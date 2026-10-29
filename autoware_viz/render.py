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

"""What the renderers share: their options, the run cache and the worker pool."""

from multiprocessing import Pool
import os

from .data import Run

RUN_CACHE = {}


def add_render_args(ap):
    g = ap.add_argument_group("what is drawn")
    g.add_argument("--map", help="the Lanelet2 .osm map (default: the one the run was made with)")
    g.add_argument("--plan", help="the label of the trajectory to follow (default: the first)")
    g.add_argument(
        "--horizon",
        type=float,
        help="seconds of each trajectory to draw (default: all of it, or 8 s when the planner "
        "leaves time_from_start empty)",
    )
    g.add_argument("--history", type=float, default=3.0, help="seconds of ego history to draw")
    g.add_argument("--title", default="", help="caption of the scene")
    g.add_argument("--header", default="", help="large text at the top, e.g. the planner's name")
    g.add_argument(
        "--subheader",
        default="",
        help="line under the header; {horizon} and {plan} are replaced",
    )
    g.add_argument("--note", default="", help='small print, e.g. "replay of a recorded drive"')


def add_selection_args(ap):
    g = ap.add_argument_group("which passes")
    g.add_argument("--start", type=float, default=0.0, help="seconds after the first pass")
    g.add_argument("--end", type=float, default=1e9, help="seconds after the first pass")
    g.add_argument("--every", type=int, default=1, help="render every n-th pass")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker processes")


def options(args, **extra):
    opts = {
        k: getattr(args, k)
        for k in ("map", "plan", "horizon", "history", "title", "header", "subheader", "note")
    }
    opts.update(extra)
    return opts


def get_run(run_dir, opts):
    key = (os.path.abspath(run_dir), opts["map"], opts["plan"])
    if key not in RUN_CACHE:
        RUN_CACHE[key] = Run(run_dir, map_path=opts["map"], plan=opts["plan"])
    return RUN_CACHE[key]


def get_frame(run_dir, index, opts):
    run = get_run(run_dir, opts)
    return run.frame(index, horizon=opts["horizon"], history=opts["history"])


def subheader(opts, frame):
    return opts["subheader"].format(horizon=f"{frame['horizon']:g}", plan=frame["plan_label"])


def render_all(fn, run_dir, out_dir, indices, opts, jobs):
    """Render the passes with fn((run_dir, out_dir, index, opts)) in a pool; print progress."""
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(run_dir, out_dir, i, opts) for i in indices]
    paths = []
    with Pool(max(1, jobs)) as pool:
        for k, path in enumerate(pool.imap(fn, tasks)):
            paths.append(path)
            if k % 50 == 0:
                print(path, flush=True)
    print(f"{len(paths)} frames")
    return paths
