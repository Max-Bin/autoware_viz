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

import math
import os

from conftest import make_run
from conftest import write_osm
import numpy as np
import pytest

from trajectory_viz.data import Run
from trajectory_viz.data import clip_plan
from trajectory_viz.data import inside
from trajectory_viz.data import lane_signal
from trajectory_viz.data import plan_from_row
from trajectory_viz.data import to_ego
from trajectory_viz.data import to_map
from trajectory_viz.lanelet_map import LaneletMap


def labels_map_ped():
    return {"label": 7, "y": 3.0, "yaw": 0.0, "l": 0.6, "w": 0.6, "h": 1.7}


def test_map(map_path):
    m = LaneletMap(map_path)
    assert sorted(m.lanes) == [100, 101]
    assert m.lane_regs == {100: [300]}
    assert len(m.light_lines[300]) == 1
    assert len(m.crosswalks) == 1 and len(m.intersections) == 1
    assert len(m.stop_lines) == 1 and len(m.borders) == 1
    assert sorted(m.near("lanes", 0.0, 0.0, 10.0)) == [100, 101]
    assert m.near("lanes", 0.0, 100.0, 10.0) == []
    # The second load comes from the cache and holds the same.
    again = LaneletMap(map_path)
    assert np.array_equal(again.lanes[100][0], m.lanes[100][0])


def test_map_without_local_coordinates(tmp_path, home):
    path = str(tmp_path / "latlon.osm")
    write_osm(path, local=False)
    with pytest.raises(ValueError, match="local_x"):
        LaneletMap(path)


def test_lane_signal():
    red, amber, green = [1, 1, 1.0], [2, 1, 1.0], [3, 1, 1.0]
    right_arrow = [3, 3, 1.0]
    assert lane_signal([]) is None and lane_signal([[0, 1, 1.0]]) is None
    assert lane_signal([green]) == "green" and lane_signal([amber]) == "amber"
    # A green right arrow beside a red circle: green for the lane turning right only.
    assert lane_signal([red, right_arrow], "right") == "green"
    assert lane_signal([red, right_arrow], "left") == "red"
    assert lane_signal([red, right_arrow]) == "red"
    assert lane_signal([[3, 6, 1.0]], "straight") == "green"  # up-right arrow
    assert lane_signal([right_arrow], "left") == "red"  # an arrow for others only


def test_inside():
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    assert inside(square) and not inside(square + 5.0)


def test_frames_round_trip():
    pose = (10.0, -3.0, 0.7)
    pts = np.array([[1.0, 2.0], [-4.0, 0.5]])
    assert np.allclose(to_ego(to_map(pts, pose), pose), pts)


def test_plan_starts_at_the_ego_and_keeps_its_times():
    row = {"frame_id": "map", "points": [[0.1 * k, k, 0.0, 0.0, 10.0] for k in range(1, 61)]}
    plan = plan_from_row(row, (0.0, 0.0, 0.0))
    assert plan["timed"]
    assert np.allclose(plan["xy"][0], [0.0, 0.0]) and plan["t"][0] == 0.0
    assert plan["duration"] == pytest.approx(6.0)
    clipped = clip_plan(plan, 2.5)
    assert clipped["t"][-1] == pytest.approx(2.5)
    assert clipped["xy"][-1][0] == pytest.approx(25.0)
    assert np.allclose(clipped["ticks"][:, 0], [10.0, 20.0])


def test_plan_without_times_and_with_points_behind():
    # Autoware's own planners: points from 5 m behind the ego, time_from_start left empty.
    row = {"frame_id": "map", "points": [[0.0, x, 0.0, 0.0, 5.0] for x in range(-5, 50)]}
    plan = plan_from_row(row, (0.0, 0.0, 0.0))
    assert not plan["timed"]
    assert plan["xy"][0][0] == pytest.approx(0.0)
    assert plan["t"][-1] == pytest.approx(49 / 5.0)


def test_frame(run_dir):
    run = Run(run_dir)
    assert len(run.passes) == 61 and run.plan_label == "plan"
    f = run.frame(run.index_at(1.0))
    assert f["time"] == pytest.approx(1.0)
    assert f["horizon"] == pytest.approx(6.0)
    assert f["speed"] == pytest.approx(10.0)
    assert f["plan"]["xy"][-1] == pytest.approx([60.0, 0.0])
    assert len(f["plan"]["ticks"]) == 6
    # The other planner, 1 m to the left.
    ((label, other),) = f["others"]
    assert label == "other" and other["xy"][-1] == pytest.approx([60.0, 1.0])
    # Driven: straight ahead, as planned.
    assert np.allclose(f["driven"][:, 1], 0.0) and f["driven"][-1][0] == pytest.approx(60.0)
    assert f["past"][0][0] == pytest.approx(-10.0)
    # The cloud's wall is 30 m ahead of the ego at the cloud's stamp, 20 ms earlier.
    assert np.allclose(f["cloud_xy"][:, 0], 29.8, atol=0.05)
    # The car 20 m ahead and the pedestrian on the crosswalk.
    labels = {o[0]: o for o in f["objects"]}
    assert labels[1][1] == pytest.approx([20.0, 0.0], abs=0.01)
    assert labels[7][1] == pytest.approx([54.0, 3.0], abs=0.01)
    assert labels[7][2] == pytest.approx(math.pi / 2)
    # Objects in the map frame need no odometry at their stamp.
    row = {"stamp_ns": 0, "frame_id": "map", "objects": [{**labels_map_ped(), "x": 64.0}]}
    (o,) = run._objects(row, f["pose"])
    assert o[1] == pytest.approx([54.0, 3.0])
    # Map: both lanes, lane 100 on the route, the red light 50 m ahead on the route.
    assert len(f["lanes"]) == 2 and len(f["route"]) == 1
    assert f["route_light"] == "red"
    ((seg, state),) = f["lights"]
    assert state == "red" and seg.mean(axis=0) == pytest.approx([50.0, 0.0])
    assert f["timing"] == [("pass", 12.0)]
    # A message without the group leaves its state as it was; later the light turns green.
    assert run.frame(run.index_at(1.4))["route_light"] == "red"
    assert run.frame(run.index_at(4.0))["route_light"] == "green"


def test_frame_options(run_dir):
    run = Run(run_dir, plan="other")
    f = run.frame(0, horizon=3.0, history=0.0)
    assert f["plan_label"] == "other" and f["others"][0][0] == "plan"
    assert f["plan"]["t"][-1] == pytest.approx(3.0) and f["driven"][-1][0] == pytest.approx(30.0)
    with pytest.raises(ValueError, match="no trajectory labelled"):
        Run(run_dir, plan="missing")


def test_trajectory_in_base_link_without_times(tmp_path, map_path):
    run = Run(make_run(str(tmp_path / "bl"), map_path, timed=False, frame_id="base_link"))
    f = run.frame(run.index_at(2.0))
    assert not f["plan"]["timed"]
    assert f["plan"]["xy"][-1] == pytest.approx([60.0, 0.0])
    assert f["horizon"] == pytest.approx(6.0)


def test_empty_trajectories_are_skipped(run_dir):
    path = os.path.join(run_dir, "trajectories.jsonl")
    with open(path, "a") as f:
        f.write('{"label": "plan", "stamp_ns": 1, "frame_id": "map", "points": []}\n')
    assert len(Run(run_dir).passes) == 61


def test_run_without_map_or_sensors(run_dir):
    for name in ("objects.jsonl", "signals.jsonl", "route.jsonl", "timing.jsonl"):
        os.remove(os.path.join(run_dir, name))
    for c in os.listdir(os.path.join(run_dir, "clouds")):
        os.remove(os.path.join(run_dir, "clouds", c))
    run = Run(run_dir)
    run.map = None
    f = run.frame(5)
    assert "cloud_xy" not in f and "lanes" not in f and f["timing"] == []
    assert f.get("route_light") is None
