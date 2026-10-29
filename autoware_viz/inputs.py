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

"""The input topics of a run, as `record` and `extract` take them."""

import os

DEFAULT_TRAJECTORY = "/planning/trajectory"


def add_input_args(ap):
    g = ap.add_argument_group("inputs (an empty topic leaves that input out)")
    g.add_argument(
        "--trajectory",
        action="append",
        metavar="[LABEL=]TOPIC",
        help="a Trajectory topic; repeat it to compare several, the first is the plan "
        f"(default: plan={DEFAULT_TRAJECTORY})",
    )
    g.add_argument("--odometry", default="/localization/kinematic_state", metavar="TOPIC")
    g.add_argument(
        "--cloud",
        default="/sensing/lidar/concatenated/pointcloud",
        metavar="TOPIC",
        help="PointCloud2 in base_link",
    )
    g.add_argument(
        "--objects",
        default="/perception/object_recognition/objects",
        metavar="TOPIC",
        help="DetectedObjects, TrackedObjects or PredictedObjects",
    )
    g.add_argument("--route", default="/planning/mission_planning/route", metavar="TOPIC")
    g.add_argument(
        "--signals",
        default="/perception/traffic_light_recognition/traffic_signals",
        metavar="TOPIC",
        help="TrafficLightGroupArray",
    )
    g.add_argument(
        "--timing",
        action="append",
        default=[],
        metavar="LABEL=TOPIC",
        help="a Float64Stamped processing time [ms] to show; repeatable",
    )
    g.add_argument("--map", help="the Lanelet2 .osm map, remembered for rendering")


def labelled(spec, default_label):
    """Split '[LABEL=]TOPIC' into (label, topic)."""
    label, sep, topic = spec.partition("=")
    return (label, topic) if sep else (default_label, spec)


def make_meta(args):
    """Return the meta.json content: which topic each input comes from."""
    specs = args.trajectory or [f"plan={DEFAULT_TRAJECTORY}"]
    trajectories = {}
    for k, spec in enumerate(specs):
        label, topic = labelled(spec, "plan" if k == 0 else spec)
        if label in trajectories:
            raise SystemExit(f"two trajectories labelled {label!r}")
        trajectories[label] = topic
    timing = dict(labelled(spec, spec) for spec in args.timing)
    meta = {
        "trajectories": trajectories,
        "plan": next(iter(trajectories)),
        "odometry": args.odometry or None,
        "cloud": args.cloud or None,
        "objects": args.objects or None,
        "route": args.route or None,
        "signals": args.signals or None,
        "timing": timing,
        "map": os.path.abspath(args.map) if args.map else None,
    }
    return meta


def topic_streams(meta):
    """Map each recorded topic to the (kind, label) entries it feeds."""
    out = {}
    for label, topic in meta["trajectories"].items():
        out.setdefault(topic, []).append(("trajectory", label))
    for kind in ("odometry", "cloud", "objects", "route", "signals"):
        if meta[kind]:
            out.setdefault(meta[kind], []).append((kind, None))
    for label, topic in meta["timing"].items():
        out.setdefault(topic, []).append(("timing", label))
    return out
