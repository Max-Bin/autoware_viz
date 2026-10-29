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
ROS messages to the rows of a run directory, and the run directory writer.

The conversions only read message fields, so they serve the live recorder and the bag
extractor alike. A run directory holds:

- meta.json: what was recorded from which topic, and the map;
- trajectories.jsonl: every trajectory, labelled, its points [t, x, y, yaw, v] in its frame;
- odometry.jsonl, objects.jsonl, route.jsonl, signals.jsonl, timing.jsonl;
- clouds/<stamp_ns>.npy: point clouds, cropped to +-100 m and thinned to the highest point of
  each 0.2 m cell, float16 [x, y, z, intensity].
"""

import json
import math
import os

import numpy as np

CLOUD_CELL_M = 0.2
CLOUD_HALF_M = 100.0
FIELD_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
# What each kind of input is, by message type.
OBJECT_TYPES = (
    "autoware_perception_msgs/msg/DetectedObjects",
    "autoware_perception_msgs/msg/TrackedObjects",
    "autoware_perception_msgs/msg/PredictedObjects",
)
TYPES = {
    "trajectory": "autoware_planning_msgs/msg/Trajectory",
    "odometry": "nav_msgs/msg/Odometry",
    "cloud": "sensor_msgs/msg/PointCloud2",
    "route": "autoware_planning_msgs/msg/LaneletRoute",
    "signals": "autoware_perception_msgs/msg/TrafficLightGroupArray",
    "timing": "autoware_internal_debug_msgs/msg/Float64Stamped",
}


def stamp_ns(t):
    return t.sec * 1_000_000_000 + t.nanosec


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def trajectory_row(msg, label):
    points = [
        [
            p.time_from_start.sec + p.time_from_start.nanosec * 1e-9,
            p.pose.position.x,
            p.pose.position.y,
            yaw_of(p.pose.orientation),
            p.longitudinal_velocity_mps,
        ]
        for p in msg.points
    ]
    return {
        "label": label,
        "stamp_ns": stamp_ns(msg.header.stamp),
        "frame_id": msg.header.frame_id,
        "points": points,
    }


def odometry_row(msg):
    p = msg.pose.pose
    return {
        "stamp_ns": stamp_ns(msg.header.stamp),
        "x": p.position.x,
        "y": p.position.y,
        "yaw": yaw_of(p.orientation),
        "vx": msg.twist.twist.linear.x,
    }


def objects_row(msg):
    objects = []
    for o in msg.objects:
        k = o.kinematics
        pwc = getattr(k, "pose_with_covariance", None) or k.initial_pose_with_covariance
        pose = pwc.pose
        best = max(o.classification, key=lambda c: c.probability) if o.classification else None
        length, width = o.shape.dimensions.x, o.shape.dimensions.y
        if (length <= 0.0 or width <= 0.0) and o.shape.footprint.points:
            fp = np.array([[q.x, q.y] for q in o.shape.footprint.points])
            length, width = np.ptp(fp[:, 0]), np.ptp(fp[:, 1])
        objects.append(
            {
                "label": int(best.label) if best else 0,
                "x": pose.position.x,
                "y": pose.position.y,
                "yaw": yaw_of(pose.orientation),
                "l": float(length),
                "w": float(width),
                "h": float(o.shape.dimensions.z),
            }
        )
    return {
        "stamp_ns": stamp_ns(msg.header.stamp),
        "frame_id": msg.header.frame_id,
        "objects": objects,
    }


def route_row(msg):
    g = msg.goal_pose
    return {
        "stamp_ns": stamp_ns(msg.header.stamp),
        "lanelets": [p.id for s in msg.segments for p in s.primitives],
        "preferred": [s.preferred_primitive.id for s in msg.segments],
        "goal": [g.position.x, g.position.y, yaw_of(g.orientation)],
    }


def signals_row(msg):
    """Each light group's elements as [color, shape, confidence] (TrafficLightElement values)."""
    return {
        "stamp_ns": stamp_ns(msg.stamp),
        "groups": {
            str(g.traffic_light_group_id): [
                [int(e.color), int(e.shape), round(float(e.confidence), 3)] for e in g.elements
            ]
            for g in msg.traffic_light_groups
        },
    }


def timing_row(msg, label):
    return {"label": label, "stamp_ns": stamp_ns(msg.stamp), "value": msg.data}


def cloud_array(msg):
    """Crop a PointCloud2 to +-100 m and keep the highest point of each 0.2 m cell."""
    fields = {f.name: f for f in msg.fields}
    names = [n for n in ("x", "y", "z", "intensity") if n in fields]
    dtype = np.dtype(
        {
            "names": names,
            "formats": ["<" + FIELD_TYPES[fields[n].datatype] for n in names],
            "offsets": [fields[n].offset for n in names],
            "itemsize": msg.point_step,
        }
    )
    pts = np.frombuffer(msg.data, dtype=dtype)
    x, y, z = (pts[n].astype(np.float32) for n in ("x", "y", "z"))
    i = pts["intensity"].astype(np.float32) if "intensity" in names else np.zeros_like(x)
    keep = (np.abs(x) < CLOUD_HALF_M) & (np.abs(y) < CLOUD_HALF_M) & (z > -3.0) & (z < 6.0)
    x, y, z, i = x[keep], y[keep], z[keep], i[keep]
    cells = int(2 * CLOUD_HALF_M / CLOUD_CELL_M) + 1
    key = np.floor((x + CLOUD_HALF_M) / CLOUD_CELL_M).astype(np.int64) * cells + np.floor(
        (y + CLOUD_HALF_M) / CLOUD_CELL_M
    ).astype(np.int64)
    order = np.lexsort((-z, key))
    first = np.ones(order.size, bool)
    first[1:] = key[order][1:] != key[order][:-1]
    sel = order[first]
    return np.stack([x[sel], y[sel], z[sel], i[sel]], axis=1).astype(np.float16)


class RunWriter:
    """Write the rows of a run directory, appending to what is there."""

    STREAMS = ("trajectories", "odometry", "objects", "route", "signals", "timing")

    def __init__(self, run_dir, meta):
        self.run_dir = run_dir
        os.makedirs(os.path.join(run_dir, "clouds"), exist_ok=True)
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        self.files = {
            name: open(os.path.join(run_dir, name + ".jsonl"), "a", buffering=1)
            for name in self.STREAMS
        }
        self.counts = dict.fromkeys(self.STREAMS + ("clouds",), 0)
        self.cloud_frames = set()

    def write(self, stream, row):
        self.files[stream].write(json.dumps(row) + "\n")
        self.counts[stream] += 1

    def handle(self, kind, label, msg):
        """Write a message of one of the inputs (inputs.topic_streams() kinds)."""
        if kind == "trajectory":
            self.write("trajectories", trajectory_row(msg, label))
        elif kind == "odometry":
            self.write("odometry", odometry_row(msg))
        elif kind == "cloud":
            self.write_cloud(msg)
        elif kind == "objects":
            self.write("objects", objects_row(msg))
        elif kind == "route":
            self.write("route", route_row(msg))
        elif kind == "signals":
            self.write("signals", signals_row(msg))
        elif kind == "timing":
            self.write("timing", timing_row(msg, label))
        else:
            raise ValueError(kind)

    def write_cloud(self, msg):
        np.save(
            os.path.join(self.run_dir, "clouds", f"{stamp_ns(msg.header.stamp)}.npy"),
            cloud_array(msg),
        )
        self.cloud_frames.add(msg.header.frame_id)
        self.counts["clouds"] += 1

    def close(self):
        for f in self.files.values():
            f.close()

    def summary(self):
        return ", ".join(f"{n} {k}" for k, n in self.counts.items())
