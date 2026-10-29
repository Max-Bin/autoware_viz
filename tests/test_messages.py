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

"""The conversions read message fields only, so stand-ins built from SimpleNamespace do."""

import argparse
import json
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from trajectory_viz import inputs
from trajectory_viz import messages


def time(t):
    return NS(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def quat(yaw):
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def pose(x, y, yaw):
    return NS(position=NS(x=x, y=y, z=0.0), orientation=quat(yaw))


def header(t, frame="map"):
    return NS(stamp=time(t), frame_id=frame)


def test_trajectory_row():
    pts = [
        NS(time_from_start=time(0.1 * k), pose=pose(k, 0.0, 0.3), longitudinal_velocity_mps=5.0)
        for k in range(1, 4)
    ]
    row = messages.trajectory_row(NS(header=header(12.5), points=pts), "plan")
    assert row["label"] == "plan" and row["stamp_ns"] == 12_500_000_000
    assert row["points"][1] == pytest.approx([0.2, 2.0, 0.0, 0.3, 5.0])


def shape(x, y, z, footprint=()):
    return NS(dimensions=NS(x=x, y=y, z=z), footprint=NS(points=list(footprint)))


def test_objects_row_detected_and_predicted():
    cls = [NS(label=1, probability=0.2), NS(label=7, probability=0.8)]
    detected = NS(
        classification=cls,
        kinematics=NS(pose_with_covariance=NS(pose=pose(3.0, 4.0, 0.5))),
        shape=shape(4.0, 2.0, 1.5),
    )
    predicted = NS(
        classification=[],
        kinematics=NS(initial_pose_with_covariance=NS(pose=pose(1.0, 2.0, 0.0))),
        shape=shape(0.0, 0.0, 1.0, [NS(x=-1.0, y=-0.5), NS(x=1.0, y=0.5)]),
    )
    row = messages.objects_row(NS(header=header(1.0, "base_link"), objects=[detected, predicted]))
    a, b = row["objects"]
    assert row["frame_id"] == "base_link"
    assert a["label"] == 7 and a["yaw"] == pytest.approx(0.5) and a["l"] == 4.0
    assert b["label"] == 0 and (b["l"], b["w"]) == (2.0, 1.0)


def element(color, shape_, confidence=1.0):
    return NS(color=color, shape=shape_, confidence=confidence)


def test_signals_row():
    group = NS(traffic_light_group_id=300, elements=[element(1, 1), element(3, 3, 0.5)])
    row = messages.signals_row(NS(stamp=time(2.0), traffic_light_groups=[group]))
    assert row == {"stamp_ns": 2_000_000_000, "groups": {"300": [[1, 1, 1.0], [3, 3, 0.5]]}}


def test_route_row():
    seg = NS(preferred_primitive=NS(id=2), primitives=[NS(id=1), NS(id=2)])
    msg = NS(header=header(0.0), segments=[seg], goal_pose=pose(5.0, 6.0, 0.0))
    row = messages.route_row(msg)
    assert row["lanelets"] == [1, 2] and row["preferred"] == [2] and row["goal"][:2] == [5.0, 6.0]


def cloud_msg(points):
    data = np.array(points, np.float32)
    fields = [
        NS(name=n, datatype=7, offset=4 * k) for k, n in enumerate(("x", "y", "z", "intensity"))
    ]
    return NS(header=header(1.0, "base_link"), fields=fields, point_step=16, data=data.tobytes())


def test_cloud_array_keeps_the_highest_point_of_each_cell():
    msg = cloud_msg(
        [
            [10.01, 5.01, 0.2, 1.0],
            [10.05, 5.05, 1.7, 2.0],  # same 0.2 m cell, higher
            [11.0, 5.0, 0.5, 3.0],
            [150.0, 0.0, 1.0, 4.0],  # beyond 100 m
            [1.0, 1.0, 9.0, 5.0],  # above 6 m
        ]
    )
    cloud = messages.cloud_array(msg).astype(np.float32)
    assert cloud.shape == (2, 4)
    assert sorted(cloud[:, 2].tolist()) == pytest.approx([0.5, 1.7], abs=0.01)


def test_meta_and_streams():
    ap = argparse.ArgumentParser()
    inputs.add_input_args(ap)
    args = ap.parse_args(
        ["--trajectory", "/a", "--trajectory", "mine=/b", "--cloud", "", "--timing", "t=/x"]
    )
    meta = inputs.make_meta(args)
    assert meta["trajectories"] == {"plan": "/a", "mine": "/b"} and meta["plan"] == "plan"
    assert meta["cloud"] is None and meta["timing"] == {"t": "/x"}
    streams = inputs.topic_streams(meta)
    assert streams["/b"] == [("trajectory", "mine")] and "/x" in streams
    json.dumps(meta)
    args = ap.parse_args([])
    assert inputs.make_meta(args)["trajectories"] == {"plan": "/planning/trajectory"}
    with pytest.raises(SystemExit):
        inputs.make_meta(ap.parse_args(["--trajectory", "a=/x", "--trajectory", "a=/y"]))


def test_writer(tmp_path):
    w = messages.RunWriter(str(tmp_path / "run"), {"plan": "plan"})
    w.handle("timing", "pass", NS(stamp=time(1.0), data=12.0))
    w.handle("cloud", None, cloud_msg([[1.0, 2.0, 1.0, 0.0]]))
    w.close()
    assert w.counts["timing"] == 1 and w.counts["clouds"] == 1
    assert (tmp_path / "run" / "clouds" / "1000000000.npy").exists()
    with pytest.raises(ValueError):
        w.handle("nonsense", None, None)
