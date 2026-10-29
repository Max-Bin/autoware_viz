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
Render scenes with the chase camera and join them into one animation.

<scenes.json> lists the scenes in order; a relative run directory is relative to the file:

    [{"run": "runs/drive1", "start": 22.5, "end": 30.5, "title": "Crosses on green"}, ...]

start and end are seconds after the run's first pass. Writes <out_dir>/reel.webp (animated,
looping) and <out_dir>/reel.mp4 (H.264), at 10 frames per second, one frame per pass. Scenes are
joined through a short fade to dark and back. A scene already rendered with the same settings is
reused from <out_dir>/sceneNN/. The drawing options apply to every scene; a scene's own "title"
replaces --title.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

from PIL import Image
from PIL import ImageColor

from . import chase
from . import render
from . import style as st


def render_scene(scene, run_dir, out_dir, opts, jobs):
    """Render one scene into out_dir unless it already holds it; return its frames."""
    spec_path = os.path.join(out_dir, "scene.json")
    spec = {"run": os.path.abspath(run_dir), "scene": scene, "options": opts}
    if os.path.exists(spec_path):
        with open(spec_path) as f:
            if json.load(f) != spec:
                shutil.rmtree(out_dir)
    if not os.path.exists(spec_path):
        run = render.get_run(run_dir, opts)
        indices = run.frames_between(scene["start"], scene["end"])
        render.render_all(chase.render_frame, run_dir, out_dir, indices, opts, jobs)
        with open(spec_path, "w") as f:
            json.dump(spec, f)
    return sorted(glob.glob(os.path.join(out_dir, "[0-9]*.png")))


def main(argv):
    ap = argparse.ArgumentParser(
        prog="trajectory-viz reel",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("scenes")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker processes")
    ap.add_argument("--fade", type=int, default=3, help="frames of fade at each cut")
    ap.add_argument("--quality", type=int, default=45, help="WebP quality, 0-100")
    ap.add_argument("--webp-width", type=int, default=960, help="width of the WebP")
    ap.add_argument("--crf", type=int, default=20, help="H.264 quality of the MP4, lower is better")
    render.add_render_args(ap)
    chase.add_chase_args(ap)
    args = ap.parse_args(argv)
    with open(args.scenes) as f:
        scenes = json.load(f)
    base = os.path.dirname(os.path.abspath(args.scenes))
    os.makedirs(args.out_dir, exist_ok=True)

    sequence = []
    for k, scene in enumerate(scenes):
        run_dir = os.path.join(base, scene["run"])
        opts = render.options(args, width=args.width, clean=args.clean)
        opts["title"] = scene.get("title", args.title)
        out_dir = os.path.join(args.out_dir, f"scene{k:02d}")
        frames = render_scene(scene, run_dir, out_dir, opts, args.jobs)
        if not frames:
            sys.exit(f"scene {k}: no passes between {scene['start']} and {scene['end']} s")
        print(f"scene {k}: {scene['run']} {scene['start']}-{scene['end']} s, {len(frames)} frames")
        sequence.append(frames)

    # Fades: the last frames of a scene darken, the first frames of the next one brighten.
    fade_dir = os.path.join(args.out_dir, "fades")
    os.makedirs(fade_dir, exist_ok=True)
    dark_rgb = ImageColor.getrgb(st.BG)
    out = []
    for k, frames in enumerate(sequence):
        for j, f in enumerate(frames):
            alpha = 1.0
            if k > 0 and j < args.fade:
                alpha = (j + 1) / (args.fade + 1)
            if k < len(sequence) - 1 and j >= len(frames) - args.fade:
                alpha = min(alpha, (len(frames) - j) / (args.fade + 1))
            if alpha < 1.0:
                im = Image.open(f).convert("RGB")
                p = os.path.join(fade_dir, f"{k:02d}_{j:05d}.png")
                Image.blend(Image.new("RGB", im.size, dark_rgb), im, alpha).save(p)
                out.append(p)
            else:
                out.append(f)

    lst = os.path.join(args.out_dir, "list.txt")
    with open(lst, "w") as fh:
        for p in out:
            fh.write(f"file '{os.path.abspath(p)}'\nduration 0.1\n")
        fh.write(f"file '{os.path.abspath(out[-1])}'\n")
    concat = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst]
    webp = os.path.join(args.out_dir, "reel.webp")
    subprocess.run(
        concat
        + ["-vf", f"fps=10,scale={args.webp_width}:-1:flags=lanczos", "-c:v", "libwebp_anim"]
        + ["-lossless", "0", "-q:v", str(args.quality), "-compression_level", "6"]
        + ["-loop", "0", webp],
        check=True,
    )
    mp4 = os.path.join(args.out_dir, "reel.mp4")
    subprocess.run(
        concat
        + ["-vf", "fps=10,format=yuv420p", "-c:v", "libx264", "-crf", str(args.crf)]
        + ["-preset", "slow", "-movflags", "+faststart", mp4],
        check=True,
    )
    print(
        f"{len(out)} frames ({len(out) / 10:.1f} s): {webp} {os.path.getsize(webp) / 2**20:.1f} MB,"
        f" {mp4} {os.path.getsize(mp4) / 2**20:.1f} MB"
    )
