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
Compose a figure of single scenes: clean chase-camera views with a caption and a speed badge.

<panels.json> lists the panels, row by row; a relative run directory is relative to the file:

    [{"run": "runs/drive1", "time": 26.9, "title": "Crosses on green"}, ...]

time is seconds after the run's first pass; the pass nearest to it is drawn.
"""

import argparse
import json
import os
import tempfile

from PIL import Image
from PIL import ImageColor

from . import chase
from . import render
from . import style as st


def main(argv):
    ap = argparse.ArgumentParser(
        prog="trajectory-viz sheet",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("panels")
    ap.add_argument("out", help="output image, e.g. scenes.png")
    ap.add_argument("--columns", type=int, default=2)
    ap.add_argument("--panel-width", type=int, default=960)
    ap.add_argument("--gap", type=int, default=8, help="pixels between panels")
    ap.add_argument(
        "--colors", type=int, default=256, help="quantize a PNG to this many colours; 0 keeps all"
    )
    render.add_render_args(ap)
    args = ap.parse_args(argv)
    with open(args.panels) as f:
        panels = json.load(f)
    base = os.path.dirname(os.path.abspath(args.panels))

    w, h, gap = args.panel_width, args.panel_width * 9 // 16, args.gap
    rows = (len(panels) + args.columns - 1) // args.columns
    sheet = Image.new(
        "RGB",
        (args.columns * w + (args.columns - 1) * gap, rows * h + (rows - 1) * gap),
        ImageColor.getrgb(st.BG),
    )
    with tempfile.TemporaryDirectory() as tmp:
        for k, panel in enumerate(panels):
            run_dir = os.path.join(base, panel["run"])
            opts = render.options(args, width=1280, clean=True)
            opts["title"] = panel.get("title", "")
            index = render.get_run(run_dir, opts).index_at(panel["time"])
            path = chase.render_frame((run_dir, tmp, index, opts))
            im = Image.open(path).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
            sheet.paste(im, ((k % args.columns) * (w + gap), (k // args.columns) * (h + gap)))
            print(f"panel {k}: {panel['run']} at {panel['time']} s")
    if args.colors and args.out.lower().endswith(".png"):
        sheet = sheet.quantize(
            colors=args.colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE
        )
    sheet.save(args.out, optimize=True)
    print(f"{args.out}: {sheet.size[0]}x{sheet.size[1]}, {os.path.getsize(args.out) // 1024} KB")
