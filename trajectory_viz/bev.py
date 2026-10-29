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
Render passes as a bird's-eye view in the frame the plan starts in, beside a side panel.

One 1280 x 720 <pass index>.png per pass: the map, route, signal states, point cloud, objects,
ego history, the trajectories and the path driven next, and a panel with the ego speed, the
signal on the route, the planned against the driven speed and the processing times.
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import render  # noqa: E402
from . import style as st  # noqa: E402

# Illustrative ego footprint around base_link (rear axle centre), not a particular vehicle.
EGO_REAR, EGO_FRONT, EGO_HALF_W = -1.0, 3.6, 0.92
VIEW = (-18.0, 62.0, 40.0)  # metres behind, ahead and to each side
FIG_W, FIG_H, DPI = 12.8, 7.2, 100
BEV_FRAC = 0.5625  # the BEV is the left 720 x 720 pixels


def view(xy):
    """Plan frame (x forward, y left) to plot coordinates (right, up)."""
    xy = np.asarray(xy)
    return np.stack([-xy[..., 1], xy[..., 0]], axis=-1)


def box_corners(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return np.stack(
        [x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]], 1
    )


def draw_bev(ax, f, view_box=VIEW, point_scale=1.0, title="", note=""):
    back, front, half = view_box
    ax.set_facecolor(st.BG)
    ax.set_xlim(-half, half)
    ax.set_ylim(back, front)
    ax.set_aspect("equal")
    ax.axis("off")
    for g in np.arange(-10 * (half // 10), half + 1, 10):
        ax.plot([g, g], [back, front], color=st.GRID, lw=0.6, zorder=0)
    for g in np.arange(10 * math.ceil(back / 10), front + 1, 10):
        ax.plot([-half, half], [g, g], color=st.GRID, lw=0.6, zorder=0)
    for g in np.arange(20, front, 20):
        ax.text(half - 1.0, g + 0.6, f"{g:.0f} m", color=st.FAINT, fontsize=7, ha="right", zorder=0)

    # Point cloud.
    if "cloud_xy" in f:
        z = f["cloud_z"]
        ground = z < 0.3
        p = view(f["cloud_xy"])
        ax.scatter(
            p[ground, 0],
            p[ground, 1],
            s=0.3 * point_scale,
            c="#26345a",
            lw=0,
            zorder=1,
            rasterized=True,
        )
        ax.scatter(
            p[~ground, 0],
            p[~ground, 1],
            s=0.9 * point_scale,
            c=np.clip(z[~ground], 0.3, 3.0),
            cmap=st.CLOUD_CMAP,
            vmin=0.3,
            vmax=3.0,
            lw=0,
            zorder=2,
            rasterized=True,
        )

    # Map.
    inter = [view(p) for p in f.get("intersections", [])]
    cross = [view(p) for p in f.get("crosswalks", [])]
    ax.add_collection(PolyCollection(inter, facecolor="#101a33", edgecolor="none", zorder=3))
    ax.add_collection(
        PolyCollection(cross, facecolor="#cbd5e1", edgecolor="none", alpha=0.16, zorder=4)
    )
    for pts in cross:
        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                fill=False,
                hatch="////",
                lw=0,
                edgecolor="#64748b",
                alpha=0.6,
                zorder=4,
            )
        )
    fills = [view(p) for p in f.get("route", [])]
    ax.add_collection(
        PolyCollection(fills, facecolor=st.ROUTE, edgecolor="none", alpha=0.13, zorder=5)
    )
    bounds = [view(b) for lane in f.get("lanes", []) for b in lane]
    ax.add_collection(LineCollection(bounds, colors=st.LANE, linewidths=0.8, zorder=6))
    borders = [view(p) for p in f.get("borders", [])]
    ax.add_collection(LineCollection(borders, colors=st.BORDER, linewidths=1.3, zorder=7))
    stops = [view(p) for p in f.get("stop_lines", [])]
    ax.add_collection(LineCollection(stops, colors="#e2e8f0", linewidths=2.0, zorder=8))
    for seg, state in f.get("lights", []):
        seg = view(seg)
        color = st.LIGHT[state]
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=4.0, solid_capstyle="butt", zorder=9)
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=11.0, alpha=0.18, zorder=9)

    # Objects.
    for label, centre, yaw, length, width, _ in f.get("objects", []):
        color = st.CLASS.get(label, st.CLASS[0])[1]
        p = view(box_corners(centre[0], centre[1], yaw, length, width))
        ax.add_patch(
            Polygon(p, closed=True, facecolor=color, alpha=0.22, edgecolor="none", zorder=10)
        )
        ax.add_patch(Polygon(p, closed=True, fill=False, edgecolor=color, lw=1.3, zorder=11))
        front_mid = (p[0] + p[1]) / 2.0
        mid = p.mean(axis=0)
        ax.plot([mid[0], front_mid[0]], [mid[1], front_mid[1]], color=color, lw=1.0, zorder=11)

    # Ego history, the path driven next, the trajectories.
    if "past" in f:
        p = view(f["past"])
        ax.plot(p[:, 0], p[:, 1], color=st.MUTED, lw=1.6, ls=(0, (1, 2)), zorder=12)
    if "driven" in f and len(f["driven"]) > 1:
        p = view(f["driven"])
        ax.plot(p[:, 0], p[:, 1], color=st.DRIVEN, lw=1.5, ls=(0, (5, 4)), alpha=0.7, zorder=13)
    for k, (_, other) in enumerate(f["others"]):
        p = view(other["xy"])
        ax.plot(p[:, 0], p[:, 1], color=st.other_color(k), lw=2.0, alpha=0.9, zorder=14)
    p = view(f["plan"]["xy"])
    ax.plot(p[:, 0], p[:, 1], color=st.PLAN, lw=10, alpha=0.16, solid_capstyle="round", zorder=14)
    ax.plot(p[:, 0], p[:, 1], color=st.PLAN, lw=3.0, solid_capstyle="round", zorder=15)
    ticks = view(f["plan"]["ticks"])
    if len(ticks):
        ax.scatter(
            ticks[:, 0], ticks[:, 1], s=16, c=st.BG, edgecolors=st.PLAN, linewidths=1.4, zorder=16
        )

    if title:
        ax.text(
            -half + 1.5,
            front - 1.5,
            title,
            color=st.TEXT,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
            zorder=20,
            bbox={"facecolor": st.BG, "edgecolor": st.PLAN, "alpha": 0.85, "pad": 6, "lw": 1.2},
        )
    ax.text(
        -half + 1.2,
        back + 1.2,
        f"{note}  ·  planning frame" if note else "planning frame",
        color=st.MUTED,
        fontsize=8,
        ha="left",
        va="bottom",
        zorder=20,
        bbox={"facecolor": st.BG, "edgecolor": "none", "alpha": 0.75, "pad": 3},
    )
    ego = np.array(
        [
            [EGO_FRONT, EGO_HALF_W],
            [EGO_FRONT, -EGO_HALF_W],
            [EGO_REAR, -EGO_HALF_W],
            [EGO_REAR, EGO_HALF_W],
        ]
    )
    pe = view(ego)
    ax.add_patch(
        FancyBboxPatch(
            (pe[:, 0].min(), pe[:, 1].min()),
            np.ptp(pe[:, 0]),
            np.ptp(pe[:, 1]),
            boxstyle="round,pad=0,rounding_size=0.45",
            facecolor="#f8fafc",
            edgecolor="none",
            zorder=17,
        )
    )
    nose = np.array([[EGO_FRONT - 0.35, 0.0], [EGO_FRONT - 1.25, 0.5], [EGO_FRONT - 1.25, -0.5]])
    ax.add_patch(Polygon(view(nose), closed=True, facecolor="#0f172a", edgecolor="none", zorder=18))


def draw_panel(fig, f, opts):
    ax = fig.add_axes([BEV_FRAC, 0.0, 1.0 - BEV_FRAC, 1.0])
    ax.set_facecolor(st.PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patches.append(
        matplotlib.patches.Rectangle(
            (BEV_FRAC, 0), 1 - BEV_FRAC, 1, transform=fig.transFigure, color=st.PANEL, zorder=-1
        )
    )
    x0 = 0.07
    if opts["header"]:
        ax.text(
            x0, 0.925, opts["header"], color=st.TEXT, fontsize=24, fontweight="bold", va="center"
        )
    sub = render.subheader(opts, f)
    if sub:
        ax.text(x0, 0.868, sub, color=st.MUTED, fontsize=10.5, va="center")
    ax.plot([x0, 0.93], [0.83, 0.83], color=st.GRID, lw=1.0)

    v = f["speed"] * 3.6
    ax.text(x0, 0.745, f"{v:3.0f}", color=st.TEXT, fontsize=40, fontweight="bold", va="center")
    ax.text(x0 + 0.215, 0.725, "km/h", color=st.MUTED, fontsize=12, va="center")
    ax.text(x0, 0.665, "ego speed", color=st.FAINT, fontsize=9, va="center")

    light = f.get("route_light")
    lx = 0.52
    if light is not None:
        ax.scatter([lx + 0.02], [0.745], s=260, c=st.LIGHT[light], zorder=3)
        ax.scatter([lx + 0.02], [0.745], s=900, c=st.LIGHT[light], alpha=0.18, zorder=2)
        ax.text(
            lx + 0.075,
            0.745,
            st.LIGHT_NAME[light],
            color=st.TEXT,
            fontsize=15,
            fontweight="bold",
            va="center",
        )
    else:
        ax.scatter([lx + 0.02], [0.745], s=260, c="none", edgecolors=st.FAINT, linewidths=1.5)
        ax.text(lx + 0.075, 0.745, "none", color=st.MUTED, fontsize=15, va="center")
    ax.text(lx, 0.665, "signal on route", color=st.FAINT, fontsize=9, va="center")

    # Speed profile: the trajectories against what was driven.
    cax = fig.add_axes([BEV_FRAC + 0.055, 0.30, 0.365, 0.29])
    cax.set_facecolor(st.PANEL)
    for side in ("top", "right"):
        cax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        cax.spines[side].set_color(st.FAINT)
    cax.tick_params(colors=st.MUTED, labelsize=8, length=3)
    cax.grid(color=st.GRID, lw=0.8)
    top = 10.0
    for k, (_, other) in enumerate(f["others"]):
        cax.plot(other["t"], other["v"] * 3.6, color=st.other_color(k), lw=1.6)
        top = max(top, other["v"].max() * 3.6)
    plan = f["plan"]
    cax.plot(plan["t"], plan["v"] * 3.6, color=st.PLAN, lw=2.4)
    top = max(top, plan["v"].max() * 3.6)
    dv = f.get("driven_speed")
    if dv is not None and len(dv[0]) > 1:
        cax.plot(dv[0], dv[1] * 3.6, color=st.DRIVEN, lw=1.4, ls=(0, (5, 4)), alpha=0.8)
        top = max(top, dv[1].max() * 3.6)
    cax.set_xlim(0, f["horizon"])
    cax.set_ylim(0, math.ceil((top * 1.15) / 10.0) * 10.0)
    cax.set_xlabel("time ahead [s]", color=st.MUTED, fontsize=9)
    cax.set_ylabel("speed [km/h]", color=st.MUTED, fontsize=9)
    cax.set_title("planned vs. driven speed", color=st.TEXT, fontsize=10.5, loc="left", pad=6)

    # Processing times.
    for k, (label, value) in enumerate(f["timing"][:2]):
        xx = x0 + k * 0.42
        ax.text(
            xx, 0.185, f"{value:.0f}", color=st.TEXT, fontsize=20, fontweight="bold", va="center"
        )
        unit_x = xx + (0.12 if value >= 100 else 0.085)
        ax.text(unit_x, 0.178, "ms", color=st.MUTED, fontsize=10, va="center")
        ax.text(xx, 0.135, label, color=st.FAINT, fontsize=9, va="center")

    # Legend, wrapping onto a second row when it runs out of width.
    has = f["has"]
    items = [(st.PLAN, "-", f["plan_label"])]
    items += [(st.other_color(k), "-", label) for k, (label, _) in enumerate(f["others"])]
    items += [(st.DRIVEN, "--", "driven next"), (st.MUTED, ":", "past")] * has["odometry"]
    items += [(st.ROUTE, "s", "route")] * has["route"]
    items += [("#38bdf8", "b", "objects")] * has["objects"]
    xx, ly = x0, 0.06
    widths = [0.08 + 0.0145 * len(label) for _, _, label in items]
    if x0 + sum(widths) > 0.97:
        ly = 0.085
    for (color, kind, label), width in zip(items, widths):
        if xx + width > 0.97 + 0.03:
            xx, ly = x0, ly - 0.045
        if kind == "s":
            ax.add_patch(
                matplotlib.patches.Rectangle((xx, ly - 0.012), 0.04, 0.024, color=color, alpha=0.35)
            )
        elif kind == "b":
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (xx, ly - 0.012), 0.04, 0.024, fill=False, edgecolor=color, lw=1.2
                )
            )
        else:
            dash = {"-": "-", "--": (0, (3, 2)), ":": (0, (1, 1.5))}[kind]
            ax.plot([xx, xx + 0.04], [ly, ly], color=color, lw=2.2, ls=dash)
        ax.text(xx + 0.05, ly, label, color=st.MUTED, fontsize=8.5, va="center")
        xx += width


def render_frame(task):
    run_dir, out_dir, index, opts = task
    f = render.get_frame(run_dir, index, opts)
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=st.BG)
    ax = fig.add_axes([0.0, 0.0, BEV_FRAC, 1.0])
    draw_bev(ax, f, title=opts["title"], note=opts["note"])
    draw_panel(fig, f, opts)
    path = os.path.join(out_dir, f"{index:05d}.png")
    fig.savefig(path, facecolor=st.BG)
    plt.close(fig)
    return path


def main(argv):
    ap = argparse.ArgumentParser(prog="trajectory-viz bev", description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    render.add_selection_args(ap)
    render.add_render_args(ap)
    args = ap.parse_args(argv)
    opts = render.options(args)
    run = render.get_run(args.run_dir, opts)
    indices = run.frames_between(args.start, args.end)[:: args.every]
    render.render_all(render_frame, args.run_dir, args.out_dir, indices, opts, args.jobs)
