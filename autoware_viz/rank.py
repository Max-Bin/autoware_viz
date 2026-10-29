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
Score every pass's plan against what was driven, then rank candidate scenes.

Per pass: the distance between where the plan puts the ego and where it drove, at 1, 2, 3 and
6 s (--horizons), and the scene: speed, heading change over the next 6 s, lateral shift without
heading change (a lane change), objects within 30 m, the signal on the route (with a map).

Candidates are 8, 10 and 12 s windows. A window ranks high when it is interesting (it turns,
changes lane, starts, stops, runs fast, or the signal changes) and the plan stays close to the
driven path at 3 s throughout. The best non-overlapping windows are printed; --json saves them
in the form `autoware-viz reel` reads.
"""

import argparse
import json
import math
import os

import numpy as np

from .data import Run
from .data import to_ego


def score_run(run, horizons):
    frames = []
    longest = max(horizons)
    for i, row in enumerate(run.passes):
        stamp = row["stamp_ns"]
        pose = run.pose_at(stamp)
        if pose is None:
            continue
        f = run.frame(i, horizon=longest, history=0.0, cloud=False)
        plan = f["plan"]
        s = {"i": i, "t": f["time"], "v": f["speed"], "err": {}}
        for h in horizons:
            later = run.pose_at(stamp + int(h * 1e9))
            if later is None or plan["t"][-1] < h - 1e-6:
                continue
            g = to_ego([later[:2]], pose)[0]
            p = np.array([np.interp(h, plan["t"], plan["xy"][:, k]) for k in (0, 1)])
            s["err"][h] = float(np.hypot(*(p - g)))
            if h == longest:
                s["lat"] = float(g[1])
        later = run.pose_at(stamp + 6_000_000_000) or run.pose_at(int(run.odom_t[-1]))
        s["dyaw6"] = math.degrees(later[2] - pose[2]) if later else 0.0
        s["light"] = f.get("route_light")
        objects = f.get("objects", [])
        s["near"] = sum(1 for o in objects if math.hypot(*o[1]) < 30.0)
        s["peds"] = sum(1 for o in objects if o[0] == 7 and math.hypot(*o[1]) < 40.0)
        frames.append(s)
    return frames


def segments(name, frames, at=3.0, lengths=(8.0, 10.0, 12.0)):
    out = []
    if not frames:
        return out
    t = np.array([f["t"] for f in frames])
    for length in lengths:
        for a in np.arange(0.0, t[-1] - length - 3.0, 0.5):
            sel = [f for f in frames if a <= f["t"] <= a + length]
            if len(sel) < length * 8:
                continue
            moving = [f for f in sel if f["v"] > 0.5 and at in f["err"]]
            if len(moving) < 0.4 * len(sel):
                continue
            e = np.array([f["err"][at] for f in moving])
            dyaw = max(abs(f["dyaw6"]) for f in sel)
            vmax = max(f["v"] for f in sel)
            vmin = min(f["v"] for f in sel)
            lights = [f["light"] for f in sel]
            light_change = any(
                x is not None and y is not None and x != y for x, y in zip(lights, lights[1:])
            )
            near = float(np.mean([f["near"] for f in sel]))
            peds = max(f["peds"] for f in sel)
            lat = max((abs(f.get("lat", 0.0)) for f in sel if abs(f["dyaw6"]) < 15), default=0.0)
            tags = []
            if dyaw > 60:
                tags.append("turn-left" if max(f["dyaw6"] for f in sel) > 60 else "turn-right")
            elif dyaw > 25:
                tags.append("curve")
            if lat > 2.5 and dyaw < 15:
                tags.append("lane-change?")
            if vmax > 11.0:
                tags.append("fast")
            if vmin < 0.2 and vmax > 4.0:
                tags.append("stop-or-start")
            if light_change:
                tags.append("signal-change")
            if peds:
                tags.append("pedestrians")
            interest = (
                min(dyaw, 90) / 90 * 2.0
                + (vmax > 11) * 1.0
                + ("stop-or-start" in tags) * 1.0
                + light_change * 0.7
                + min(near, 10) / 10 * 0.6
                + min(peds, 4) / 4 * 0.5
                + ("lane-change?" in tags) * 1.5
            )
            median = float(np.median(e))
            p90 = float(np.percentile(e, 90))
            out.append(
                {
                    "run": name,
                    "start": round(float(a), 1),
                    "end": round(float(a + length), 1),
                    "tags": tags,
                    "interest": round(float(interest), 2),
                    "error_median": round(median, 2),
                    "error_p90": round(p90, 2),
                    "error_max": round(float(e.max()), 2),
                    "vmax_kmh": round(vmax * 3.6),
                    "turn_deg": round(dyaw),
                    "score": round(float(interest - 1.5 * median - 0.5 * p90), 2),
                }
            )
    return out


def main(argv):
    ap = argparse.ArgumentParser(
        prog="autoware-viz rank",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--json", help="write the best scenes here, as a reel's scene list")
    ap.add_argument("--top", type=int, default=25, help="how many scenes to print and save")
    ap.add_argument("--horizons", default="1,2,3,6", help="seconds ahead to measure the error at")
    ap.add_argument("--map", help="the Lanelet2 .osm map (default: the one the run was made with)")
    ap.add_argument("--plan", help="the label of the trajectory to score (default: the first)")
    args = ap.parse_args(argv)
    horizons = sorted(float(h) for h in args.horizons.split(","))
    at = 3.0 if 3.0 in horizons else horizons[len(horizons) // 2]
    allseg, summary = [], {}
    for run_dir in args.runs:
        name = os.path.basename(os.path.normpath(run_dir))
        frames = score_run(Run(run_dir, map_path=args.map, plan=args.plan), horizons)
        moving = [f for f in frames if f["v"] > 0.5]
        for h in horizons:
            summary.setdefault(h, []).extend(f["err"][h] for f in moving if h in f["err"])
        print(f"{name}: {len(frames)} passes with odometry, {len(moving)} moving")
        for seg in segments(name, frames, at):
            seg["run"] = run_dir
            allseg.append(seg)
    for h, e in summary.items():
        if e:
            print(
                f"error at {h:g} s over the moving passes: median {np.median(e):.2f} m, "
                f"p90 {np.percentile(e, 90):.2f} m"
            )
    allseg.sort(key=lambda s: -s["score"])
    chosen = []
    for s in allseg:
        if not any(
            c["run"] == s["run"] and s["start"] < c["end"] and c["start"] < s["end"] for c in chosen
        ):
            chosen.append(s)
    chosen = chosen[: args.top]
    print(f"\nbest scenes (start-end [s]; error at {at:g} s):")
    for s in chosen:
        print(
            f"  {os.path.basename(os.path.normpath(s['run']))} {s['start']:5.1f}-{s['end']:<5.1f} "
            f"score {s['score']:5.2f}  "
            f"error median {s['error_median']:.2f} p90 {s['error_p90']:.2f} m  "
            f"{s['vmax_kmh']:3d} km/h  turn {s['turn_deg']:3d} deg  {' '.join(s['tags'])}"
        )
    if args.json:
        base = os.path.dirname(os.path.abspath(args.json))
        scenes = [
            {
                "run": os.path.relpath(os.path.abspath(s["run"]), base),
                "start": s["start"],
                "end": s["end"],
                "title": " ".join(s["tags"]),
            }
            for s in chosen
        ]
        with open(args.json, "w") as f:
            json.dump(scenes, f, indent=1)
