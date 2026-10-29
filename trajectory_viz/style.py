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

"""Colours shared by the renderers."""

from matplotlib.colors import LinearSegmentedColormap

BG = "#070b17"
PANEL = "#0c1224"
GRID = "#131b30"
TEXT = "#e8edf7"
MUTED = "#8a96b3"
FAINT = "#4a5675"
LANE = "#2c3858"
BORDER = "#5d6a8a"
ROUTE = "#14d6a0"
PLAN = "#3cf0b4"
DRIVEN = "#f4f6fb"
LIGHT = {"green": "#22c55e", "amber": "#f5a524", "red": "#ef4444"}
LIGHT_NAME = {"green": "GREEN", "amber": "AMBER", "red": "RED"}
# Further trajectories, in the order they were given.
OTHERS = ["#f472b6", "#fbbf24", "#c084fc", "#60a5fa", "#fb923c"]
CLASS = {  # autoware_perception_msgs ObjectClassification labels
    0: ("unknown", "#9aa5bd"),
    1: ("car", "#38bdf8"),
    2: ("truck", "#a78bfa"),
    3: ("bus", "#a78bfa"),
    4: ("trailer", "#a78bfa"),
    5: ("motorcycle", "#facc15"),
    6: ("bicycle", "#facc15"),
    7: ("pedestrian", "#fb923c"),
}
CLOUD_CMAP = LinearSegmentedColormap.from_list(
    "cloud", ["#2f4f8f", "#3b82f6", "#5cc8ff", "#e0f6ff"]
)


def other_color(k):
    return OTHERS[k % len(OTHERS)]
