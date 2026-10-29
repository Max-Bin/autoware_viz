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
Record a run directory live, from the topics of a running stack or a playing bag.

Stop it with Ctrl-C. Needs a sourced ROS 2 workspace with the Autoware messages.
"""

import argparse

from .inputs import add_input_args
from .inputs import make_meta
from .inputs import topic_streams
from .messages import OBJECT_TYPES
from .messages import RunWriter
from .messages import TYPES


def main(argv):
    ap = argparse.ArgumentParser(prog="autoware-viz record", description=__doc__)
    ap.add_argument("run_dir")
    add_input_args(ap)
    import rclpy

    args = ap.parse_args(rclpy.utilities.remove_ros_args(["record"] + argv)[1:])
    meta = make_meta(args)

    import rclpy.executors
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy
    from rclpy.qos import HistoryPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message

    qos = {
        "cloud": QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5
        ),
        "route": QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        ),
    }
    default_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=50
    )

    class Recorder(Node):
        def __init__(self, writer, streams):
            super().__init__("autoware_viz_recorder")
            self.writer = writer
            self.pending = {}  # topics whose message type is found by discovery
            self.subs = []
            for topic, entries in streams.items():
                kinds = {kind for kind, _ in entries}
                if kinds == {"objects"}:
                    self.pending[topic] = entries
                else:
                    self.subscribe(topic, TYPES[entries[0][0]], entries)
            if self.pending:
                self.create_timer(1.0, self.discover)

        def subscribe(self, topic, type_name, entries):
            def callback(msg):
                for kind, label in entries:
                    self.writer.handle(kind, label, msg)

            profile = qos.get(entries[0][0], default_qos)
            self.subs.append(
                self.create_subscription(get_message(type_name), topic, callback, profile)
            )
            self.get_logger().info(f"recording {topic} ({type_name})")

        def discover(self):
            for topic, types in self.get_topic_names_and_types():
                if topic in self.pending:
                    found = [t for t in types if t in OBJECT_TYPES]
                    if found:
                        self.subscribe(topic, found[0], self.pending.pop(topic))

    rclpy.init()
    writer = RunWriter(args.run_dir, meta)
    node = Recorder(writer, topic_streams(meta))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except rclpy._rclpy_pybind11.RCLError as e:
        # SIGINT can invalidate the context while spin() builds its wait set.
        if "context is not valid" not in str(e):
            raise
    finally:
        writer.close()
        print(f"recorded {writer.summary()}", flush=True)
