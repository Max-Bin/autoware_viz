# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A run directory, and what one frame of it shows.

A frame is one message of the plan's trajectory topic. Everything in it is put in the frame the
plan starts in: base_link at the trajectory's stamp, x forward and y left, with the ego pose
interpolated from the odometry.
"""

import bisect
import glob
import json
import math
import os

import numpy as np

from .lanelet_map import LaneletMap

# autoware_perception_msgs/TrafficLightElement values.
COLORS = {1: "red", 2: "amber", 3: "green"}
GREEN, CIRCLE = 3, 1
# The arrows that let a lane with each turn_direction go, as Autoware's traffic light module
# reads them: left, right and up arrows, and the up-left and up-right combinations.
ARROWS = {"left": {2, 5}, "right": {3, 6}, "straight": {4, 5, 6}}
MAP_RADIUS_M = 150.0


def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def lane_signal(elements, turn_direction=None):
    """
    Return what a light group means for a lane: green, amber, red, or None when it is unknown.

    A green circle lets every lane go and a green arrow the lanes turning its way; otherwise the
    circle's colour holds, and lit arrows for other directions mean stop.
    """
    lit = [(c, shape) for c, shape, _ in elements if c in COLORS]
    if not lit:
        return None
    if (GREEN, CIRCLE) in lit:
        return "green"
    if any(c == GREEN and shape in ARROWS.get(turn_direction, ()) for c, shape in lit):
        return "green"
    circles = [c for c, shape in lit if shape == CIRCLE]
    return COLORS[circles[0]] if circles else "red"


def inside(poly, x=0.0, y=0.0):
    """Return whether (x, y) is inside a polygon [N, 2] (even-odd rule)."""
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    crosses = (py > y) != (qy > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = px + (y - py) * (qx - px) / (qy - py)
    return bool(np.count_nonzero(crosses & (x < xs)) % 2)


def to_ego(xy, pose):
    """Move map-frame points into the body frame at pose (x, y, yaw)."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rx, ry = xy[:, 0] - x, xy[:, 1] - y
    return np.stack([c * rx + s * ry, -s * rx + c * ry], axis=1)


def to_map(xy, pose):
    """Move body-frame points at pose (x, y, yaw) into the map frame."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * xy[:, 0] - s * xy[:, 1], y + s * xy[:, 0] + c * xy[:, 1]], axis=1)


def trajectory_times(xy, t, v):
    """
    Return each point's time, from time_from_start where the planner filled it in.

    Where it did not (all zero, or not increasing), the time is integrated along the path from
    the point speeds, with a floor of 0.1 m/s; the first point is then at 0 s.
    """
    if len(t) > 1 and t[-1] > t[0] and np.all(np.diff(t) >= -1e-6):
        return t, True
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = np.maximum(0.5 * (np.abs(v[:-1]) + np.abs(v[1:])), 0.1)
    return np.concatenate([[0.0], np.cumsum(ds / speed)]), False


def plan_from_row(row, pose, body_pose=None):
    """
    Return a trajectory in the frame at pose: points xy, times t, speeds v, its duration.

    The points before the one nearest to the ego are dropped, and the ego position is put in
    front with time 0 when the trajectory starts ahead of it. body_pose is the pose of a
    trajectory published in base_link.
    """
    pts = np.asarray(row["points"], np.float64).reshape(-1, 5)
    if not len(pts):
        return None
    xy = pts[:, 1:3]
    if row.get("frame_id") == "base_link" and body_pose is not None:
        xy = to_map(xy, body_pose)
    xy = to_ego(xy, pose)
    k = int(np.argmin(np.hypot(xy[:, 0], xy[:, 1])))
    xy, t, v = xy[k:], pts[k:, 0] - (pts[k, 0] if k else 0.0), pts[k:, 4]
    t, timed = trajectory_times(xy, t, v)
    gap = float(np.hypot(*xy[0]))
    if gap > 0.05:
        lead = t[0] if timed and t[0] > 0 else gap / max(abs(v[0]), 0.1)
        xy = np.concatenate([[[0.0, 0.0]], xy])
        t = np.concatenate([[0.0], t + (0.0 if timed and t[0] > 0 else lead)])
        v = np.concatenate([[v[0]], v])
    return {"xy": xy, "t": t, "v": v, "duration": float(t[-1]), "timed": timed}


def clip_plan(plan, horizon):
    """Return the part of a plan up to horizon seconds, with ticks at whole seconds."""
    t = plan["t"]
    keep = t <= horizon + 1e-6
    xy, tt, v = plan["xy"][keep], t[keep], plan["v"][keep]
    if t[-1] > horizon and len(tt) and tt[-1] < horizon:
        end = [np.interp(horizon, t, plan["xy"][:, i]) for i in (0, 1)]
        xy = np.concatenate([xy, [end]])
        tt = np.append(tt, horizon)
        v = np.append(v, np.interp(horizon, t, plan["v"]))
    seconds = np.arange(1.0, min(horizon, t[-1]) + 1e-6)
    ticks = np.stack([np.interp(seconds, t, plan["xy"][:, i]) for i in (0, 1)], axis=1)
    return {**plan, "xy": xy, "t": tt, "v": v, "ticks": ticks.reshape(-1, 2)}


class Run:
    def __init__(self, run_dir, map_path=None, plan=None):
        self.dir = run_dir
        with open(os.path.join(run_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.trajectories = {}
        for row in _jsonl(os.path.join(run_dir, "trajectories.jsonl")):
            if row["points"]:  # a planner may publish empty trajectories, e.g. while it waits
                self.trajectories.setdefault(row["label"], []).append(row)
        for rows in self.trajectories.values():
            rows.sort(key=lambda r: r["stamp_ns"])
        self.plan_label = plan or self.meta["plan"]
        self.passes = self.trajectories.get(self.plan_label, [])
        if not self.passes:
            have = ", ".join(self.trajectories) or "none"
            raise ValueError(f"{run_dir}: no trajectory labelled {self.plan_label!r} ({have})")
        self.traj_stamps = {k: [r["stamp_ns"] for r in v] for k, v in self.trajectories.items()}
        self.t0 = self.passes[0]["stamp_ns"]

        odom = sorted(_jsonl(os.path.join(run_dir, "odometry.jsonl")), key=lambda o: o["stamp_ns"])
        self.odom_t = np.array([o["stamp_ns"] for o in odom], np.int64)
        self.odom_xy = np.array([[o["x"], o["y"]] for o in odom]).reshape(-1, 2)
        self.odom_yaw = np.unwrap(np.array([o["yaw"] for o in odom], np.float64))
        self.odom_v = np.array([o["vx"] for o in odom], np.float64)

        self.objects, self.objects_t = self._stream("objects")
        # Per light group, its states over time: messages may each carry only some groups.
        self.signal_groups = {}
        for row in self._stream("signals")[0]:
            for gid, elements in row["groups"].items():
                self.signal_groups.setdefault(int(gid), ([], []))
                self.signal_groups[int(gid)][0].append(row["stamp_ns"])
                self.signal_groups[int(gid)][1].append(elements)
        self.routes, self.routes_t = self._stream("route")
        self.timing = {}
        for row in _jsonl(os.path.join(run_dir, "timing.jsonl")):
            self.timing.setdefault(row["label"], []).append((row["stamp_ns"], row["value"]))
        for label in self.timing:
            self.timing[label].sort()
        self.timing_t = {k: [s for s, _ in v] for k, v in self.timing.items()}
        clouds = glob.glob(os.path.join(run_dir, "clouds", "*.npy"))
        self.cloud_t = sorted(int(os.path.basename(c)[:-4]) for c in clouds)

        path = map_path or self.meta.get("map")
        self.map = LaneletMap(path) if path else None

    def _stream(self, name):
        rows = sorted(_jsonl(os.path.join(self.dir, name + ".jsonl")), key=lambda r: r["stamp_ns"])
        return rows, [r["stamp_ns"] for r in rows]

    def time_of(self, index):
        return (self.passes[index]["stamp_ns"] - self.t0) / 1e9

    def frames_between(self, start, end):
        """Return the indices of the passes from start to end seconds after the first one."""
        return [i for i in range(len(self.passes)) if start <= self.time_of(i) <= end]

    def index_at(self, time):
        """Return the index of the pass nearest to time seconds after the first one."""
        return min(range(len(self.passes)), key=lambda i: abs(self.time_of(i) - time))

    def pose_at(self, stamp_ns):
        """Interpolated (x, y, yaw) from the odometry, or None outside it."""
        if len(self.odom_t) < 2 or not self.odom_t[0] <= stamp_ns <= self.odom_t[-1]:
            return None
        i = min(max(int(np.searchsorted(self.odom_t, stamp_ns)), 1), len(self.odom_t) - 1)
        a = (stamp_ns - self.odom_t[i - 1]) / max(1, self.odom_t[i] - self.odom_t[i - 1])
        xy = self.odom_xy[i - 1] + a * (self.odom_xy[i] - self.odom_xy[i - 1])
        yaw = self.odom_yaw[i - 1] + a * (self.odom_yaw[i] - self.odom_yaw[i - 1])
        return float(xy[0]), float(xy[1]), float(yaw)

    def speed_at(self, stamp_ns):
        if not len(self.odom_t):
            return None
        return float(np.interp(stamp_ns, self.odom_t, self.odom_v))

    @staticmethod
    def latest(stamps, stamp_ns, max_age_ns, lead_ns=0):
        """Return the index of the last stamp at most lead_ns after stamp_ns and not too old."""
        i = bisect.bisect_right(stamps, stamp_ns + lead_ns) - 1
        if i >= 0 and stamp_ns - stamps[i] <= max_age_ns:
            return i
        return None

    @staticmethod
    def nearest(stamps, stamp_ns, tolerance_ns):
        i = bisect.bisect_left(stamps, stamp_ns)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(stamps) and abs(stamps[j] - stamp_ns) <= tolerance_ns:
                if best is None or abs(stamps[j] - stamp_ns) < abs(stamps[best] - stamp_ns):
                    best = j
        return best

    def frame(self, index, horizon=None, history=3.0, cloud=True):
        """Assemble what pass `index` shows, in the frame the plan starts in."""
        row = self.passes[index]
        stamp = row["stamp_ns"]
        pose = self.pose_at(stamp)
        if pose is None:  # no odometry here: the ego is where the trajectory starts
            p = row["points"][0]
            pose = (p[1], p[2], p[3])
        plan = plan_from_row(row, pose, pose)
        if horizon is None:
            horizon = plan["duration"] if plan["timed"] else min(plan["duration"], 8.0)
        horizon = max(float(horizon), 0.1)
        speed = self.speed_at(stamp)
        f = {
            "index": index,
            "time": self.time_of(index),
            "pose": pose,
            "speed": speed if speed is not None else float(plan["v"][0]),
            "horizon": horizon,
            "plan_label": self.plan_label,
            "plan": clip_plan(plan, horizon),
            "others": [],
            # What the run has at all, so that legends do not change from frame to frame.
            "has": {
                "cloud": bool(self.cloud_t),
                "objects": bool(self.objects),
                "map": self.map is not None,
                "route": bool(self.routes),
                "signals": bool(self.signal_groups),
                "odometry": len(self.odom_t) > 1,
            },
        }
        for label, rows in self.trajectories.items():
            if label == self.plan_label:
                continue
            j = self.nearest(self.traj_stamps[label], stamp, 150_000_000)
            if j is not None:
                other = plan_from_row(rows[j], pose, self.pose_at(rows[j]["stamp_ns"]))
                if other is not None:
                    f["others"].append((label, clip_plan(other, horizon)))

        if len(self.odom_t):
            past = (self.odom_t >= stamp - int(history * 1e9)) & (self.odom_t <= stamp)
            if past.sum() > 1:
                f["past"] = to_ego(self.odom_xy[past], pose)
            ahead = (self.odom_t > stamp) & (self.odom_t <= stamp + int(horizon * 1e9))
            if ahead.any():
                f["driven"] = np.concatenate([[[0.0, 0.0]], to_ego(self.odom_xy[ahead], pose)])
                f["driven_speed"] = ((self.odom_t[ahead] - stamp) / 1e9, self.odom_v[ahead])

        if cloud:
            j = self.latest(self.cloud_t, stamp, 300_000_000, lead_ns=50_000_000)
            src = self.pose_at(self.cloud_t[j]) if j is not None else None
            if src is not None:
                pts = np.load(os.path.join(self.dir, "clouds", f"{self.cloud_t[j]}.npy"))
                f["cloud_xy"] = to_ego(to_map(pts[:, :2].astype(np.float64), src), pose)
                f["cloud_z"] = pts[:, 2].astype(np.float64)

        j = self.latest(self.objects_t, stamp, 300_000_000, lead_ns=50_000_000)
        if j is not None:
            f["objects"] = self._objects(self.objects[j], pose)

        f["timing"] = []
        for label, series in self.timing.items():
            k = self.nearest(self.timing_t[label], stamp, 150_000_000)
            if k is not None:
                f["timing"].append((label, series[k][1]))

        if self.map is not None:
            self._map_layers(f, stamp, pose)
        return f

    def _objects(self, row, pose):
        """Return the objects as (label, centre, yaw, length, width, height) in the frame."""
        if row["frame_id"] == "map":
            src = None
        else:  # base_link at the objects' stamp
            src = self.pose_at(row["stamp_ns"])
            if src is None:
                return []
        out = []
        for o in row["objects"]:
            centre = np.array([[o["x"], o["y"]]])
            yaw = o["yaw"]
            if src is not None:
                centre = to_map(centre, src)
                yaw += src[2]
            out.append((o["label"], to_ego(centre, pose)[0], yaw - pose[2], o["l"], o["w"], o["h"]))
        return out

    def _map_layers(self, f, stamp, pose):
        m = self.map
        x, y = pose[0], pose[1]
        lanes = m.near("lanes", x, y, MAP_RADIUS_M)
        f["lanes"] = [(to_ego(m.lanes[i][0], pose), to_ego(m.lanes[i][1], pose)) for i in lanes]
        f["crosswalks"] = [to_ego(p, pose) for p in m.near("crosswalks", x, y, MAP_RADIUS_M)]
        f["intersections"] = [to_ego(p, pose) for p in m.near("intersections", x, y, MAP_RADIUS_M)]
        f["stop_lines"] = [to_ego(p, pose) for p in m.near("stop_lines", x, y, MAP_RADIUS_M)]
        f["borders"] = [to_ego(p, pose) for p in m.near("borders", x, y, MAP_RADIUS_M)]

        route = set()
        if self.routes:
            j = self.latest(self.routes_t, stamp, 1 << 62)
            route = set(self.routes[j if j is not None else 0]["lanelets"])
        on_route = [i for i in lanes if i in route]
        f["route"] = [
            to_ego(np.concatenate([m.lanes[i][0], m.lanes[i][1][::-1]]), pose) for i in on_route
        ]

        groups, regs = {}, {}  # traffic light -> its elements, and the lanes near it controls
        for i in lanes:
            for reg in m.lane_regs.get(i, ()):
                if reg not in groups and reg in self.signal_groups:
                    stamps, states = self.signal_groups[reg]
                    k = self.latest(stamps, stamp, 1_500_000_000)
                    groups[reg] = states[k] if k is not None else None
                if groups.get(reg) is not None:
                    regs.setdefault(reg, []).append(i)
        lights, ahead, current = [], [], None
        for reg, controlled in regs.items():
            # A light means what it means for the lanes it controls: the route's lane through it
            # if there is one, else a lane that turns (whose arrows apply), else any of them.
            on = [i for i in controlled if i in route]
            pool = on or controlled
            lane = next((i for i in pool if i in m.lane_turns), pool[0])
            state = lane_signal(groups[reg], m.lane_turns.get(lane))
            if state is None:
                continue
            if m.light_lines.get(reg):
                segs = [to_ego(s, pose) for s in m.light_lines[reg]]
            else:  # no stop line: across the end of each lane
                ends = [np.stack([m.lanes[i][0][-1], m.lanes[i][1][-1]]) for i in controlled]
                segs = [to_ego(e, pose) for e in ends]
            lights.append((segs, state, bool(on)))
            if not on:
                continue
            for seg in segs:
                mid = seg.mean(axis=0)
                if mid[0] > -2.0:
                    ahead.append((float(np.hypot(*mid)), state))
            if any(inside(f["route"][on_route.index(i)]) for i in on):
                current = state
        # Lights of the route last, on top of others sharing their stop line.
        f["lights"] = [
            (seg, st) for segs, st, _ in sorted(lights, key=lambda x: x[2]) for seg in segs
        ]
        # The signal on the route: the one of the lane the ego is on, else the next one ahead.
        f["route_light"] = current or (min(ahead)[1] if ahead else None)
