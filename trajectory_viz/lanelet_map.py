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
The parts of a Lanelet2 map that are drawn, read straight from its .osm file.

Points are taken from the local_x / local_y tags that Autoware's maps carry, which are the
coordinates of the map frame; no projection library is needed. A parsed map is cached under
~/.cache/trajectory_viz, keyed by the file's path, size and modification time.
"""

import hashlib
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np

LANE_SUBTYPES = {"road", "highway", "road_shoulder", "bicycle_lane"}
CACHE_VERSION = 2


class LaneletMap:
    """Lanes, crosswalks, intersection areas, stop lines, road borders and traffic lights."""

    def __init__(self, path):
        self.path = path
        data = _load_cached(path)
        self.lanes = data["lanes"]  # lanelet id -> (left [N, 2], right [M, 2])
        self.lane_regs = data["lane_regs"]  # lanelet id -> traffic-light regulatory element ids
        self.lane_turns = data["lane_turns"]  # lanelet id -> turn_direction (left, right, straight)
        self.crosswalks = data["crosswalks"]  # polygons [N, 2]
        self.intersections = data["intersections"]  # polygons [N, 2]
        self.stop_lines = data["stop_lines"]  # polylines [N, 2]
        self.borders = data["borders"]  # polylines [N, 2]
        self.light_lines = data["light_lines"]  # regulatory element id -> stop lines [N, 2]
        self._index = {}
        for name in ("crosswalks", "intersections", "stop_lines", "borders"):
            self._index[name] = _bounds(getattr(self, name))
        self.lane_ids = list(self.lanes)
        self._index["lanes"] = _bounds(
            [np.concatenate(self.lanes[i]) for i in self.lane_ids], self.lane_ids
        )

    def near(self, name, x, y, radius):
        """Return the elements of a layer whose bounding box comes within radius of (x, y)."""
        lo, hi, keys = self._index[name]
        if not len(lo):
            return []
        dx = np.maximum(np.maximum(lo[:, 0] - x, x - hi[:, 0]), 0.0)
        dy = np.maximum(np.maximum(lo[:, 1] - y, y - hi[:, 1]), 0.0)
        hit = np.nonzero(dx * dx + dy * dy <= radius * radius)[0]
        if name == "lanes":
            return [keys[i] for i in hit]
        items = getattr(self, name)
        return [items[i] for i in hit]


def _bounds(items, keys=None):
    lo = np.array([p.min(axis=0) for p in items]) if items else np.zeros((0, 2))
    hi = np.array([p.max(axis=0) for p in items]) if items else np.zeros((0, 2))
    return lo, hi, keys


def _load_cached(path):
    st = os.stat(path)
    key = f"{CACHE_VERSION}:{os.path.abspath(path)}:{st.st_size}:{st.st_mtime_ns}"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "trajectory_viz")
    cache = os.path.join(cache_dir, hashlib.sha1(key.encode()).hexdigest() + ".pkl")
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # rebuilt below
    data = parse_osm(path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cache}.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
    except OSError:
        pass  # a read-only home: parse every time
    return data


def parse_osm(path):
    """Parse an Autoware Lanelet2 .osm file into the arrays LaneletMap holds."""
    nodes, ways, relations = {}, {}, {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "node":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if "local_x" in tags and "local_y" in tags:
                nodes[int(el.get("id"))] = (float(tags["local_x"]), float(tags["local_y"]))
            el.clear()
        elif el.tag == "way":
            ways[int(el.get("id"))] = (
                [int(n.get("ref")) for n in el.findall("nd")],
                {t.get("k"): t.get("v") for t in el.findall("tag")},
            )
            el.clear()
        elif el.tag == "relation":
            relations[int(el.get("id"))] = (
                [(m.get("type"), m.get("role"), int(m.get("ref"))) for m in el.findall("member")],
                {t.get("k"): t.get("v") for t in el.findall("tag")},
            )
            el.clear()
    if not nodes:
        raise ValueError(
            f"{path}: no node has local_x / local_y tags; trajectory_viz reads the map frame "
            "coordinates from them, as Autoware's Lanelet2 maps carry them"
        )

    def line(way_id):
        refs = ways.get(way_id, ([], {}))[0]
        pts = [nodes[r] for r in refs if r in nodes]
        return np.array(pts, np.float64) if len(pts) >= 2 else None

    def ways_of(tag_type):
        out = []
        for way_id, (_, tags) in ways.items():
            if tags.get("type") == tag_type:
                pts = line(way_id)
                if pts is not None:
                    out.append(pts)
        return out

    lanes, lane_regs, lane_turns, crosswalks, light_lines = {}, {}, {}, [], {}
    for rel_id, (members, tags) in relations.items():
        kind = tags.get("type")
        if kind == "lanelet":
            left = next((r for t, role, r in members if t == "way" and role == "left"), None)
            right = next((r for t, role, r in members if t == "way" and role == "right"), None)
            left, right = line(left), line(right)
            if left is None or right is None:
                continue
            subtype = tags.get("subtype")
            if subtype == "crosswalk":
                crosswalks.append(np.concatenate([left, right[::-1]]))
            elif subtype in LANE_SUBTYPES:
                lanes[rel_id] = (left, right)
                if tags.get("turn_direction") in ("left", "right", "straight"):
                    lane_turns[rel_id] = tags["turn_direction"]
                regs = [r for t, role, r in members if role == "regulatory_element"]
                if regs:
                    lane_regs[rel_id] = regs
        elif kind == "regulatory_element" and tags.get("subtype") == "traffic_light":
            stops = [line(r) for t, role, r in members if t == "way" and role == "ref_line"]
            light_lines[rel_id] = [s for s in stops if s is not None]
    lane_regs = {
        lane: [r for r in regs if r in light_lines]
        for lane, regs in lane_regs.items()
        if any(r in light_lines for r in regs)
    }
    return {
        "lanes": lanes,
        "lane_regs": lane_regs,
        "lane_turns": lane_turns,
        "crosswalks": crosswalks,
        "intersections": [p for p in ways_of("intersection_area") if len(p) >= 3],
        "stop_lines": ways_of("stop_line"),
        "borders": ways_of("road_border"),
        "light_lines": light_lines,
    }
