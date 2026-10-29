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
trajectory-viz: render a planner's trajectories over a recorded drive.

commands:
  record   record a run directory live from ROS 2 topics (needs ROS 2)
  extract  extract a run directory from rosbag2 recordings (needs ROS 2)
  rank     score the plans against what was driven and rank candidate scenes
  bev      render passes as a bird's-eye view beside a side panel
  chase    render passes from a chase camera, with minimap, speed chart and legend
  reel     render a list of scenes and join them into an animated WebP and an MP4
  sheet    compose single scenes into one captioned figure

Run `trajectory-viz <command> --help` for a command's options.
"""

import importlib
import sys

from . import __version__

COMMANDS = ("record", "extract", "rank", "bev", "chase", "reel", "sheet")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if argv[0] == "--version":
        print(__version__)
        return 0
    if argv[0] not in COMMANDS:
        print(f"unknown command {argv[0]!r}; one of: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    importlib.import_module(f"trajectory_viz.{argv[0]}").main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
