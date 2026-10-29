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
Extract a run directory from rosbag2 recordings (sqlite3 or mcap).

Give the bags of one drive in order; a route or a map recorded only in the first one then
holds for the rest. Needs a sourced ROS 2 workspace with the Autoware messages.
"""

import argparse
import os
import sys

from .inputs import add_input_args
from .inputs import make_meta
from .inputs import topic_streams
from .messages import OBJECT_TYPES
from .messages import RunWriter
from .messages import TYPES


def storage_of(path):
    """Return (uri, storage id) of a bag directory or a single bag file."""
    if os.path.isdir(path):
        meta = os.path.join(path, "metadata.yaml")
        if os.path.exists(meta):
            with open(meta) as f:
                for line in f:
                    if line.strip().startswith("storage_identifier:"):
                        return path, line.split(":", 1)[1].strip()
        files = sorted(os.listdir(path))
        if any(f.endswith(".mcap") for f in files):
            return path, "mcap"
        return path, "sqlite3"
    return path, "mcap" if path.endswith(".mcap") else "sqlite3"


def main(argv):
    ap = argparse.ArgumentParser(prog="trajectory-viz extract", description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("bags", nargs="+", help="bag directories or files of one drive, in order")
    add_input_args(ap)
    args = ap.parse_args(argv)
    meta = make_meta(args)
    streams = topic_streams(meta)

    from rclpy.serialization import deserialize_message
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message

    writer = RunWriter(args.run_dir, meta)
    failed = {}
    try:
        for bag in args.bags:
            uri, storage_id = storage_of(bag)
            reader = rosbag2_py.SequentialReader()
            reader.open(
                rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
                rosbag2_py.ConverterOptions("cdr", "cdr"),
            )
            types = {t.name: t.type for t in reader.get_all_topics_and_types()}
            classes = {}
            for topic, entries in streams.items():
                if topic not in types:
                    continue
                kind = entries[0][0]
                expected = OBJECT_TYPES if kind == "objects" else (TYPES[kind],)
                if types[topic] not in expected:
                    print(f"{bag}: {topic} is {types[topic]}, not {' or '.join(expected)}")
                    continue
                classes[topic] = get_message(types[topic])
            missing = sorted(set(streams) - set(classes))
            print(
                f"{bag}: reading {len(classes)} topics"
                + (f"; not in it: {missing}" * bool(missing))
            )
            if not classes:
                continue
            reader.set_filter(rosbag2_py.StorageFilter(topics=list(classes)))
            while reader.has_next():
                topic, data, _ = reader.read_next()
                try:
                    msg = deserialize_message(data, classes[topic])
                except Exception as e:  # e.g. recorded with another message definition
                    if topic not in failed:
                        print(f"{bag}: cannot read {topic}: {e}", file=sys.stderr)
                    failed[topic] = failed.get(topic, 0) + 1
                    continue
                for kind, label in streams[topic]:
                    writer.handle(kind, label, msg)
    finally:
        writer.close()
    for topic, n in failed.items():
        print(f"skipped {n} unreadable messages of {topic}")
    print(f"extracted {writer.summary()}")
