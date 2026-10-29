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

import json
import os
import shutil

from PIL import Image
import pytest

from autoware_viz.__main__ import main


def pngs(path):
    return sorted(f for f in os.listdir(path) if f.endswith(".png"))


@pytest.mark.parametrize("command", ["bev", "chase"])
def test_render_frames(run_dir, tmp_path, command):
    out = str(tmp_path / command)
    args = [command, run_dir, out, "--start", "1", "--end", "1.25", "--jobs", "2"]
    main(args + ["--title", "Stops for a red light", "--header", "Planner", "--note", "test"])
    frames = pngs(out)
    assert frames == ["00010.png", "00011.png", "00012.png"]
    assert Image.open(os.path.join(out, frames[0])).size == (1280, 720)


def test_chase_options(run_dir, tmp_path):
    out = str(tmp_path / "chase")
    main(["chase", run_dir, out, "--end", "0", "--width", "640", "--clean", "--horizon", "2"])
    assert Image.open(os.path.join(out, "00000.png")).size == (640, 360)
    main(
        ["chase", run_dir, out, "--end", "0", "--plan", "other", "--subheader", "{plan} {horizon}"]
    )


def test_sheet(run_dir, tmp_path):
    panels = tmp_path / "panels.json"
    panels.write_text(
        json.dumps([{"run": "run", "time": 1.0, "title": "a"}, {"run": "run", "time": 3.0}])
    )
    out = str(tmp_path / "sheet.png")
    main(["sheet", str(panels), out, "--panel-width", "320"])
    assert Image.open(out).size == (2 * 320 + 8, 180)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_reel(run_dir, tmp_path):
    scenes = tmp_path / "scenes.json"
    scenes.write_text(
        json.dumps(
            [
                {"run": "run", "start": 0.0, "end": 0.5, "title": "one"},
                {"run": "run", "start": 3.0, "end": 3.5, "title": "two"},
            ]
        )
    )
    out = str(tmp_path / "reel")
    main(["reel", str(scenes), out, "--jobs", "2", "--width", "320"])
    assert os.path.getsize(os.path.join(out, "reel.webp")) > 0
    assert os.path.getsize(os.path.join(out, "reel.mp4")) > 0
    # Rendered scenes are reused; a changed one is rendered again.
    first = os.path.getmtime(os.path.join(out, "scene00", "00000.png"))
    main(["reel", str(scenes), out, "--jobs", "2", "--width", "320"])
    assert os.path.getmtime(os.path.join(out, "scene00", "00000.png")) == first
    main(["reel", str(scenes), out, "--jobs", "2", "--width", "320", "--horizon", "3"])
    assert os.path.getmtime(os.path.join(out, "scene00", "00000.png")) != first


def test_rank(run_dir, tmp_path, capsys):
    out = tmp_path / "ranked.json"
    main(["rank", run_dir, "--json", str(out)])
    text = capsys.readouterr().out
    assert "error at 3 s" in text
    assert json.loads(out.read_text()) == [] or "start" in json.loads(out.read_text())[0]


def test_cli(capsys):
    assert main([]) == 0
    assert "commands:" in capsys.readouterr().out
    assert main(["nonsense"]) == 2
