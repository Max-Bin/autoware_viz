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

"""A small synthetic map and run: a straight two-lane road with a signalized stop line."""

import json
import math
import os

import numpy as np
import pytest

# Two lanes along +x from x = 0 to 200, 3.5 m wide: lane 100 spans y -1.75 .. 1.75 and lane 101
# y 1.75 .. 5.25. A stop line crosses lane 100 at x = 60, governed by traffic light 300; a
# crosswalk spans x 62 .. 66 and an intersection area x 70 .. 90.
OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm>
{nodes}
  <way id="10"><nd ref="1"/><nd ref="2"/><tag k="type" v="line_thin"/></way>
  <way id="11"><nd ref="3"/><nd ref="4"/><tag k="type" v="line_thin"/></way>
  <way id="12"><nd ref="5"/><nd ref="6"/><tag k="type" v="road_border"/></way>
  <way id="13"><nd ref="7"/><nd ref="8"/><tag k="type" v="stop_line"/></way>
  <way id="14"><nd ref="9"/><nd ref="10"/><tag k="type" v="pedestrian_marking"/></way>
  <way id="15"><nd ref="11"/><nd ref="12"/><tag k="type" v="pedestrian_marking"/></way>
  <way id="16"><nd ref="13"/><nd ref="14"/><nd ref="15"/><nd ref="16"/><nd ref="13"/>
    <tag k="type" v="intersection_area"/></way>
  <relation id="100">
    <member type="way" role="left" ref="11"/><member type="way" role="right" ref="10"/>
    <member type="relation" role="regulatory_element" ref="300"/>
    <tag k="type" v="lanelet"/><tag k="subtype" v="road"/>
  </relation>
  <relation id="101">
    <member type="way" role="left" ref="12"/><member type="way" role="right" ref="11"/>
    <tag k="type" v="lanelet"/><tag k="subtype" v="road"/>
  </relation>
  <relation id="102">
    <member type="way" role="left" ref="14"/><member type="way" role="right" ref="15"/>
    <tag k="type" v="lanelet"/><tag k="subtype" v="crosswalk"/>
  </relation>
  <relation id="300">
    <member type="way" role="ref_line" ref="13"/>
    <tag k="type" v="regulatory_element"/><tag k="subtype" v="traffic_light"/>
  </relation>
</osm>
"""
NODES = {
    1: (0.0, -1.75),
    2: (200.0, -1.75),
    3: (0.0, 1.75),
    4: (200.0, 1.75),
    5: (0.0, 5.25),
    6: (200.0, 5.25),
    7: (60.0, -1.75),
    8: (60.0, 1.75),
    9: (62.0, -1.75),
    10: (62.0, 5.25),
    11: (66.0, -1.75),
    12: (66.0, 5.25),
    13: (70.0, -10.0),
    14: (90.0, -10.0),
    15: (90.0, 10.0),
    16: (70.0, 10.0),
}
SPEED = 10.0  # m/s along +x
T0 = 1_000_000_000_000_000_000


def write_osm(path, local=True):
    nodes = "\n".join(
        f'  <node id="{i}" lat="0" lon="0">'
        + (f'<tag k="local_x" v="{x}"/><tag k="local_y" v="{y}"/>' if local else "")
        + "</node>"
        for i, (x, y) in NODES.items()
    )
    with open(path, "w") as f:
        f.write(OSM.format(nodes=nodes))


def stamp(t):
    return T0 + int(round(t * 1e9))


def jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_run(run_dir, map_path, timed=True, frame_id="map"):
    """Drive along lane 100 at 10 m/s for 12 s; plan every 0.1 s, 6 s ahead."""
    os.makedirs(os.path.join(run_dir, "clouds"))
    meta = {
        "trajectories": {"plan": "/planning/trajectory", "other": "/other/trajectory"},
        "plan": "plan",
        "odometry": "/odom",
        "cloud": "/cloud",
        "objects": "/objects",
        "route": "/route",
        "signals": "/signals",
        "timing": {"pass": "/pass"},
        "map": map_path,
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    odom = [
        {"stamp_ns": stamp(t), "x": SPEED * t, "y": 0.0, "yaw": 0.0, "vx": SPEED}
        for t in np.arange(0.0, 12.0 + 1e-9, 0.02)
    ]
    jsonl(os.path.join(run_dir, "odometry.jsonl"), odom)
    trajectories = []
    for t in np.arange(0.0, 6.0 + 1e-9, 0.1):
        x0 = SPEED * t
        points = []
        for k in range(1, 61):  # 0.1 .. 6.0 s ahead
            dt = 0.1 * k
            points.append([dt if timed else 0.0, x0 + SPEED * dt, 0.0, 0.0, SPEED])
        if frame_id == "base_link":
            points = [[p[0], p[1] - x0, p[2], p[3], p[4]] for p in points]
        trajectories.append(
            {"label": "plan", "stamp_ns": stamp(t), "frame_id": frame_id, "points": points}
        )
        # Another planner, 1 m to the left.
        other = [[p[0], p[1], p[2] + 1.0, p[3], p[4]] for p in points]
        trajectories.append(
            {"label": "other", "stamp_ns": stamp(t), "frame_id": frame_id, "points": other}
        )
    jsonl(os.path.join(run_dir, "trajectories.jsonl"), trajectories)
    # In base_link: a car 20 m ahead, and a pedestrian crossing at x = 64, y = 3 of the map.
    objects = []
    for t in np.arange(0.0, 6.0 + 1e-9, 0.1):
        car = {"label": 1, "x": 20.0, "y": 0.0, "yaw": 0.0, "l": 4.5, "w": 1.8, "h": 1.5}
        ped = {"label": 7, "x": 64.0 - SPEED * t, "y": 3.0, "yaw": math.pi / 2}
        ped.update({"l": 0.6, "w": 0.6, "h": 1.7})
        objects.append({"stamp_ns": stamp(t), "frame_id": "base_link", "objects": [car, ped]})
    jsonl(os.path.join(run_dir, "objects.jsonl"), objects)
    jsonl(
        os.path.join(run_dir, "route.jsonl"),
        [{"stamp_ns": stamp(-1.0), "lanelets": [100], "preferred": [100], "goal": [190, 0, 0]}],
    )
    jsonl(
        os.path.join(run_dir, "signals.jsonl"),
        [
            {"stamp_ns": stamp(0.0), "groups": {"300": [[1, 1, 1.0]]}},  # red circle
            {"stamp_ns": stamp(1.2), "groups": {"999": [[3, 1, 1.0]]}},  # another source's group
            {"stamp_ns": stamp(3.0), "groups": {"300": [[3, 1, 1.0]]}},  # green circle
        ],
    )
    jsonl(
        os.path.join(run_dir, "timing.jsonl"),
        [{"label": "pass", "stamp_ns": stamp(t), "value": 12.0} for t in np.arange(0, 6.01, 0.1)],
    )
    # A wall 30 m ahead, 2 m tall, in base_link at each cloud stamp.
    for t in np.arange(0.0, 6.0 + 1e-9, 0.1):
        ys = np.linspace(-10, 10, 50)
        zs = np.linspace(0.0, 2.0, 5)
        pts = np.array([[30.0, y, z, 0.0] for y in ys for z in zs], np.float16)
        np.save(os.path.join(run_dir, "clouds", f"{stamp(t) - 20_000_000}.npy"), pts)
    return run_dir


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def map_path(tmp_path, home):
    path = str(tmp_path / "map.osm")
    write_osm(path)
    return path


@pytest.fixture
def run_dir(tmp_path, map_path):
    return make_run(str(tmp_path / "run"), map_path)
