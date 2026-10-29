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
Render passes from a chase camera, with a bird's-eye minimap, a speed chart and a legend.

The scene is projected through a pinhole camera behind and above the ego and drawn back to
front with matplotlib: the map, route, signal states, point cloud, objects, the trajectories
and the path driven next, in the frame the plan starts in. --clean leaves only the scene, the
caption and a speed badge. One <pass index>.png per pass, 1280 x 720 unless --width says
otherwise.
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import bev  # noqa: E402
from . import render  # noqa: E402
from . import style as st  # noqa: E402

CAM = np.array([-15.0, 0.0, 8.0])
LOOK = np.array([24.0, 0.0, 0.0])
FOV_X = math.radians(66.0)
NEAR = 0.5
FOG_START, FOG_END = 45.0, 110.0
SKY_TOP, SKY_HORIZON = "#04060d", "#0d1530"


class Camera:
    def __init__(self, w, h, clean):
        f = LOOK - CAM
        self.f = f / np.linalg.norm(f)
        right = np.cross(self.f, [0.0, 0.0, 1.0])
        self.r = right / np.linalg.norm(right)
        self.u = np.cross(self.r, self.f)
        self.fx = (w / 2.0) / math.tan(FOV_X / 2.0)
        # The image centre sits in the area left of the overlay column, not the frame's.
        self.cx = w / 2.0 if clean else (w - 340.0 * w / 1280.0) / 2.0 + 10.0 * w / 1280.0
        self.cy = h / 2.0

    def to_cam(self, p):
        d = np.asarray(p, np.float64) - CAM
        return np.stack([d @ self.r, d @ self.u, d @ self.f], axis=-1)

    def project_cam(self, c):
        return np.stack(
            [
                self.cx + self.fx * c[..., 0] / c[..., 2],
                self.cy - self.fx * c[..., 1] / c[..., 2],
            ],
            axis=-1,
        )

    def horizon_y(self):
        far = self.to_cam(np.array([1e5, 0.0, 0.0]))
        return float(self.project_cam(far)[1])


def ground(xy, z=0.0):
    xy = np.asarray(xy, np.float64)
    return np.concatenate([xy, np.full((*xy.shape[:-1], 1), z)], axis=-1)


def clip_polyline(cam_pts):
    """Split a polyline (camera coordinates) into pieces in front of the near plane."""
    pieces, cur = [], []
    for a, b in zip(cam_pts[:-1], cam_pts[1:]):
        ina, inb = a[2] > NEAR, b[2] > NEAR
        if ina and inb:
            if not cur:
                cur.append(a)
            cur.append(b)
        elif ina or inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            m = a + t * (b - a)
            if ina:
                if not cur:
                    cur.append(a)
                cur.append(m)
                pieces.append(np.array(cur))
                cur = []
            else:
                cur = [m, b]
        else:
            if cur:
                pieces.append(np.array(cur))
                cur = []
    if len(cur) > 1:
        pieces.append(np.array(cur))
    return pieces


def clip_polygon(cam_pts):
    """Clip a polygon (camera coordinates) to the near plane; None when nothing is left."""
    out = []
    n = len(cam_pts)
    for k in range(n):
        a, b = cam_pts[k], cam_pts[(k + 1) % n]
        ina, inb = a[2] > NEAR, b[2] > NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else None


def fog(depth):
    return np.clip(1.0 - (np.asarray(depth) - FOG_START) / (FOG_END - FOG_START), 0.0, 1.0)


def add_lines(ax, cam, lines3d, color, lw, alpha=1.0, z=1, segment_fog=True):
    segs, cols = [], []
    base = np.array(to_rgba(color))
    for line in lines3d:
        c = cam.to_cam(line)
        for piece in clip_polyline(c):
            p = cam.project_cam(piece)
            if segment_fog:
                d = 0.5 * (piece[:-1, 2] + piece[1:, 2])
                for k in range(len(p) - 1):
                    segs.append(p[k : k + 2])
                    cols.append([*base[:3], base[3] * alpha * fog(d[k])])
            else:
                segs.append(p)
                cols.append([*base[:3], base[3] * alpha * fog(piece[:, 2].mean())])
    if segs:
        ax.add_collection(
            LineCollection(segs, colors=cols, linewidths=lw, zorder=z, capstyle="round")
        )


def add_polys(ax, cam, polys3d, color, alpha, z=1, edge=None):
    verts, cols = [], []
    base = np.array(to_rgba(color))
    for poly in polys3d:
        c = clip_polygon(cam.to_cam(poly))
        if c is None:
            continue
        verts.append(cam.project_cam(c))
        cols.append([*base[:3], alpha * fog(c[:, 2].mean())])
    if verts:
        ax.add_collection(
            PolyCollection(
                verts, facecolors=cols, edgecolors=edge or "none", linewidths=0.6, zorder=z
            )
        )


def box_faces(x, y, yaw, length, width, z0, z1):
    corners = bev.box_corners(x, y, yaw, length, width)
    lo = ground(corners, z0)
    hi = ground(corners, z1)
    faces = [hi]  # top
    for k in range(4):
        j = (k + 1) % 4
        faces.append(np.array([lo[k], lo[j], hi[j], hi[k]]))
    return faces, lo, hi


def draw_box(
    ax, cam, x, y, yaw, length, width, height, color, z, face_alpha=0.22, lw=1.2, top_alpha=None
):
    faces, lo, hi = box_faces(x, y, yaw, length, width, 0.0, height)
    order = sorted(range(len(faces)), key=lambda k: -cam.to_cam(faces[k]).mean(axis=0)[2])
    base = np.array(to_rgba(color))
    for k in order:
        c = clip_polygon(cam.to_cam(faces[k]))
        if c is None:
            continue
        a = (top_alpha if (k == 0 and top_alpha is not None) else face_alpha) * fog(c[:, 2].mean())
        ax.add_patch(
            plt.Polygon(
                cam.project_cam(c),
                closed=True,
                facecolor=[*base[:3], a],
                edgecolor="none",
                zorder=z,
            )
        )
    edges = [np.array([lo[k], lo[(k + 1) % 4]]) for k in range(4)]
    edges += [np.array([hi[k], hi[(k + 1) % 4]]) for k in range(4)]
    edges += [np.array([lo[k], hi[k]]) for k in range(4)]
    add_lines(ax, cam, edges, color, lw, z=z + 0.1, segment_fog=False)


# A low-poly sedan around base_link (rear axle): side profile (x, z) from the rear bumper over
# the roof to the front bumper, and the half width at each profile point.
CAR_PROFILE = [
    (-1.00, 0.28, 0.90),
    (-1.00, 0.80, 0.90),
    (-0.75, 0.92, 0.88),
    (-0.25, 1.40, 0.74),
    (1.35, 1.44, 0.74),
    (2.05, 0.98, 0.86),
    (3.45, 0.84, 0.90),
    (3.60, 0.30, 0.90),
]
CAR_GLASS = {2, 4}  # rear window and windshield strips
LIGHT_DIR = np.array([-0.4, 0.35, 0.85]) / np.linalg.norm([-0.4, 0.35, 0.85])


def draw_ego(ax, cam, z):
    shadow = ground(bev.box_corners(1.3, 0, 0, 4.9, 2.0), 0.01)
    add_polys(ax, cam, [shadow], "#000000", 0.32, z=z - 0.5)
    left = np.array([[x, hw, zz] for x, zz, hw in CAR_PROFILE])
    right = np.array([[x, -hw, zz] for x, zz, hw in CAR_PROFILE])
    faces = [(left, "#dfe5ef"), (right, "#dfe5ef")]
    for k in range(len(CAR_PROFILE) - 1):
        quad = np.array([left[k], left[k + 1], right[k + 1], right[k]])
        faces.append((quad, "#1d2a47" if k in CAR_GLASS else "#eef2f8"))
    faces.append((np.array([left[0], left[-1], right[-1], right[0]]), "#1b2333"))  # bottom
    # Back to front, each face shaded by its angle to the light.
    for poly, color in sorted(faces, key=lambda f: -cam.to_cam(f[0]).mean(axis=0)[2]):
        c = clip_polygon(cam.to_cam(poly))
        if c is None:
            continue
        n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        n = n / max(np.linalg.norm(n), 1e-9)
        shade = 0.62 + 0.38 * abs(float(n @ LIGHT_DIR))
        rgb = np.clip(np.array(to_rgba(color)[:3]) * shade, 0, 1)
        ax.add_patch(
            plt.Polygon(
                cam.project_cam(c),
                closed=True,
                facecolor=rgb,
                edgecolor=np.clip(rgb * 1.08, 0, 1),
                linewidth=0.4,
                zorder=z,
            )
        )
    # Tail lights.
    for y in (0.62, -0.62):
        tl = np.array(
            [
                [-1.01, y - 0.18, 0.62],
                [-1.01, y + 0.18, 0.62],
                [-1.01, y + 0.18, 0.72],
                [-1.01, y - 0.18, 0.72],
            ]
        )
        c = clip_polygon(cam.to_cam(tl))
        if c is not None:
            ax.add_patch(
                plt.Polygon(
                    cam.project_cam(c),
                    closed=True,
                    facecolor="#ff4d5e",
                    edgecolor="none",
                    zorder=z + 0.2,
                )
            )


def draw_plan_ribbon(ax, cam, plan, z):
    pts = plan["xy"]
    if len(pts) < 2:
        return
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    # A glow: three widths, the narrowest the most opaque, fading along the plan.
    for half, alpha, zz in ((1.6, 0.09, z), (0.95, 0.20, z + 0.1), (0.45, 0.9, z + 0.2)):
        left, right = pts + n * half, pts - n * half
        polys = []
        for k in range(len(pts) - 1):
            polys.append(ground(np.array([left[k], left[k + 1], right[k + 1], right[k]]), 0.05))
        verts, cols = [], []
        base = np.array(to_rgba(st.PLAN))
        for k, poly in enumerate(polys):
            c = clip_polygon(cam.to_cam(poly))
            if c is None:
                continue
            fade = 1.0 - 0.55 * k / len(polys)
            verts.append(cam.project_cam(c))
            cols.append([*base[:3], alpha * fade * fog(c[:, 2].mean())])
        ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors="none", zorder=zz))
    if not len(plan["ticks"]):
        return
    ticks = ground(plan["ticks"], 0.08)  # one per second
    c = cam.to_cam(ticks)
    ok = c[:, 2] > NEAR
    p = cam.project_cam(c[ok])
    ax.scatter(p[:, 0], p[:, 1], s=26, c=st.BG, edgecolors=st.PLAN, linewidths=1.6, zorder=z + 0.3)


def draw_scene(ax, cam, f, w, h):
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    hy = cam.horizon_y()
    grad = np.linspace(0, 1, 256)[:, None]
    sky = matplotlib.colors.LinearSegmentedColormap.from_list("sky", [SKY_TOP, SKY_HORIZON])
    ax.imshow(grad, extent=(0, w, hy, 0), cmap=sky, aspect="auto", zorder=0)
    ax.add_patch(plt.Rectangle((0, hy), w, h - hy, color=st.BG, zorder=0))
    glow = matplotlib.colors.LinearSegmentedColormap.from_list(
        "glow", [to_rgba("#111a38", 0.85), (0, 0, 0, 0)]
    )
    ax.imshow(
        np.linspace(0, 1, 64)[:, None],
        extent=(0, w, hy + 60, hy),
        cmap=glow,
        aspect="auto",
        zorder=0.5,
    )

    grid = [ground(np.array([[x, -60.0], [x, 60.0]])) for x in np.arange(-10, 121, 10)]
    grid += [ground(np.array([[-15.0, y], [120.0, y]])) for y in np.arange(-60, 61, 10)]
    grid = [np.linspace(g[0], g[1], 28) for g in grid]
    add_lines(ax, cam, grid, "#141c35", 0.5, z=1)

    # Map.
    add_polys(ax, cam, [ground(p) for p in f.get("intersections", [])], "#0f1a36", 0.9, z=2)
    add_polys(ax, cam, [ground(p) for p in f.get("crosswalks", [])], "#cbd5e1", 0.20, z=3)
    add_polys(ax, cam, [ground(p, 0.01) for p in f.get("route", [])], st.ROUTE, 0.16, z=4)
    bounds = [ground(b, 0.02) for lane in f.get("lanes", []) for b in lane]
    add_lines(ax, cam, bounds, "#34436a", 0.75, z=5)
    add_lines(ax, cam, [ground(p, 0.03) for p in f.get("borders", [])], "#7183aa", 1.4, z=6)
    add_lines(ax, cam, [ground(p, 0.03) for p in f.get("stop_lines", [])], "#e2e8f0", 2.4, z=7)

    # Point cloud, far points first.
    if "cloud_xy" in f:
        xy, zz = f["cloud_xy"], f["cloud_z"]
        keep = (xy[:, 0] > -20) & (xy[:, 0] < 115) & (np.abs(xy[:, 1]) < 70)
        c = cam.to_cam(np.concatenate([xy[keep], zz[keep, None]], axis=1))
        zk = zz[keep]
        ok = c[:, 2] > NEAR
        c, zk = c[ok], zk[ok]
        order = np.argsort(-c[:, 2])
        c, zk = c[order], zk[order]
        p = cam.project_cam(c)
        inside = (p[:, 0] > -5) & (p[:, 0] < w + 5) & (p[:, 1] > -5) & (p[:, 1] < h + 5)
        p, c, zk = p[inside], c[inside], zk[inside]
        depth = c[:, 2]
        g = zk < 0.3
        size = np.clip(55.0 / depth, 0.15, 3.0)
        col = st.CLOUD_CMAP(np.clip((zk - 0.3) / 2.7, 0, 1))
        col[:, 3] = fog(depth) * np.where(g, 0.35, 0.95)
        col[g, :3] = to_rgba("#2a3a66")[:3]
        ax.scatter(
            p[:, 0], p[:, 1], s=size * np.where(g, 0.6, 1.0), c=col, lw=0, zorder=8, rasterized=True
        )

    # Signal states: a bar on the stop line and a lamp above its middle.
    for seg, state in f.get("lights", []):
        color = st.LIGHT[state]
        bar = [ground(seg, 0.05)]
        add_lines(ax, cam, bar, color, 5.0, z=9)
        add_lines(ax, cam, bar, color, 16.0, alpha=0.18, z=9)
        c = cam.to_cam(ground(seg.mean(axis=0), 4.2)[None])
        if c[0, 2] > NEAR:
            p = cam.project_cam(c)[0]
            k = fog(c[0, 2])
            sz = float(np.clip(5200.0 / c[0, 2], 20, 260))
            ax.scatter([p[0]], [p[1]], s=sz * 5, c=[to_rgba(color, 0.12 * k)], lw=0, zorder=9)
            ax.scatter([p[0]], [p[1]], s=sz, c=[to_rgba(color, 0.95 * k)], lw=0, zorder=9)

    # Objects, far boxes first.
    objects = sorted(f.get("objects", []), key=lambda o: -cam.to_cam(ground(o[1], 0.0)[None])[0, 2])
    for label, centre, yaw, length, width, height in objects:
        if not (-20 < centre[0] < 110 and abs(centre[1]) < 60):
            continue
        color = st.CLASS.get(label, st.CLASS[0])[1]
        draw_box(
            ax,
            cam,
            centre[0],
            centre[1],
            yaw,
            length,
            width,
            max(height, 1.0),
            color,
            z=10,
            face_alpha=0.16,
            top_alpha=0.30,
        )

    # Ego history, the path driven next (dashed), the trajectories and the ego.
    if "past" in f:
        add_lines(ax, cam, [ground(f["past"], 0.04)], st.MUTED, 1.4, alpha=0.6, z=11)
    if "driven" in f and len(f["driven"]) > 1:
        pts = ground(f["driven"], 0.06)
        dashes = []
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))])
        for s0 in np.arange(0, cum[-1], 2.4):
            sel = (cum >= s0) & (cum <= s0 + 1.3)
            if sel.sum() >= 2:
                dashes.append(pts[sel])
        add_lines(ax, cam, dashes, st.DRIVEN, 1.8, alpha=0.75, z=12)
    for k, (_, other) in enumerate(f["others"]):
        add_lines(ax, cam, [ground(other["xy"], 0.07)], st.other_color(k), 2.2, alpha=0.95, z=12.5)
    draw_plan_ribbon(ax, cam, f["plan"], z=13)
    draw_ego(ax, cam, z=20)


def card(fig, x, y, w, h):
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.012",
            transform=fig.transFigure,
            facecolor=to_rgba(st.PANEL, 0.86),
            edgecolor=to_rgba("#26304d", 1.0),
            lw=1.0,
            zorder=30,
        )
    )


def draw_minimap(fig, f, x, y, w, h):
    ax = fig.add_axes([x, y, w, h], zorder=31)
    bev.draw_bev(ax, f, view_box=(-15.0, 55.0, 35.0), point_scale=0.3)
    for child in list(ax.texts):
        child.remove()
    ax.set_facecolor(st.BG)


def draw_clean_overlay(fig, f, opts):
    fx, fy = 1.0 / 1280.0, 1.0 / 720.0
    if opts["title"]:
        fig.text(
            28 * fx,
            1 - 50 * fy,
            opts["title"],
            color=st.TEXT,
            fontsize=22,
            fontweight="bold",
            zorder=40,
            bbox={
                "facecolor": to_rgba(st.BG, 0.8),
                "edgecolor": st.PLAN,
                "lw": 1.6,
                "boxstyle": "round,pad=0.45",
            },
        )
    light = f.get("route_light")
    x_text = 64 if light is not None else 28
    fig.text(
        x_text * fx,
        34 * fy,
        f"{f['speed'] * 3.6:.0f} km/h",
        color=st.TEXT,
        fontsize=20,
        fontweight="bold",
        zorder=40,
        bbox={"facecolor": to_rgba(st.BG, 0.8), "edgecolor": "none", "boxstyle": "round,pad=0.4"},
    )
    if light is not None:
        fig.patches.append(
            matplotlib.patches.Ellipse(
                (40 * fx, 46 * fy),
                24 * fx,
                24 * fy,
                transform=fig.transFigure,
                facecolor=st.LIGHT[light],
                zorder=41,
            )
        )


def draw_overlay(fig, f, opts):
    W, H = 1280.0, 720.0  # layout units; fonts scale with the dpi alike
    fx, fy = 1.0 / W, 1.0 / H
    horizon = f["horizon"]
    # Header, sub-header, note and caption.
    if opts["header"]:
        fig.text(
            28 * fx,
            1 - 46 * fy,
            opts["header"],
            color=st.TEXT,
            fontsize=24,
            fontweight="bold",
            zorder=40,
        )
    sub = render.subheader(opts, f)
    if sub:
        fig.text(30 * fx, 1 - 74 * fy, sub, color=st.MUTED, fontsize=10.5, zorder=40)
    if opts["note"]:
        fig.text(30 * fx, 1 - 94 * fy, opts["note"], color=st.FAINT, fontsize=8.5, zorder=40)
    if opts["title"]:
        fig.text(
            (W - 320) / 2 * fx,
            1 - 128 * fy,
            opts["title"],
            color=st.TEXT,
            fontsize=14,
            fontweight="bold",
            ha="center",
            zorder=40,
            bbox={
                "facecolor": to_rgba(st.BG, 0.8),
                "edgecolor": st.PLAN,
                "lw": 1.2,
                "boxstyle": "round,pad=0.45",
            },
        )

    # Right column: minimap card.
    cw = 300 * fx
    cx = 1 - cw - 20 * fx
    card(fig, cx, 1 - (20 + 300) * fy, cw, 300 * fy)
    draw_minimap(fig, f, cx + 6 * fx, 1 - (20 + 294) * fy, cw - 12 * fx, 288 * fy)
    fig.text(
        cx + 12 * fx,
        1 - 42 * fy,
        "bird's-eye view",
        color=st.MUTED,
        fontsize=8.5,
        zorder=40,
        bbox={"facecolor": to_rgba(st.BG, 0.85), "edgecolor": "none", "pad": 2.5},
    )

    # Right column: speed card, the trajectories against what was driven.
    card(fig, cx, 1 - (20 + 300 + 14 + 200) * fy, cw, 200 * fy)
    cax = fig.add_axes(
        [cx + 46 * fx, 1 - (20 + 300 + 14 + 168) * fy, cw - 62 * fx, 130 * fy], zorder=31
    )
    cax.set_facecolor("none")
    for side in ("top", "right"):
        cax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        cax.spines[side].set_color(st.FAINT)
    cax.tick_params(colors=st.MUTED, labelsize=7.5, length=2.5)
    cax.grid(color=st.GRID, lw=0.7)
    top = 10.0
    for k, (_, other) in enumerate(f["others"]):
        cax.plot(other["t"], other["v"] * 3.6, color=st.other_color(k), lw=1.5)
        top = max(top, other["v"].max() * 3.6)
    plan = f["plan"]
    cax.plot(plan["t"], plan["v"] * 3.6, color=st.PLAN, lw=2.2)
    top = max(top, plan["v"].max() * 3.6)
    dv = f.get("driven_speed")
    if dv is not None and len(dv[0]) > 1:
        cax.plot(dv[0], dv[1] * 3.6, color=st.DRIVEN, lw=1.3, ls=(0, (4, 3)), alpha=0.8)
        top = max(top, dv[1].max() * 3.6)
    cax.set_xlim(0, horizon)
    cax.set_ylim(0, math.ceil(top * 1.15 / 10.0) * 10.0)
    cax.set_xlabel("time ahead [s]", color=st.MUTED, fontsize=8, labelpad=1)
    cax.set_ylabel("km/h", color=st.MUTED, fontsize=8, labelpad=1)
    fig.text(
        cx + 12 * fx,
        1 - (20 + 300 + 14 + 18) * fy,
        "planned vs. driven speed",
        color=st.TEXT,
        fontsize=9.5,
        zorder=40,
    )

    # Bottom-left: speed, signal on the route, processing times.
    card(fig, 20 * fx, 20 * fy, 380 * fx, 112 * fy)
    speed = f"{f['speed'] * 3.6:.0f}"
    fig.text(40 * fx, 62 * fy, speed, color=st.TEXT, fontsize=36, fontweight="bold", zorder=40)
    fig.text(
        (40 + 34 * len(speed) + 10) * fx, 64 * fy, "km/h", color=st.MUTED, fontsize=11, zorder=40
    )
    fig.text(40 * fx, 36 * fy, "ego speed", color=st.FAINT, fontsize=8.5, zorder=40)
    light = f.get("route_light")
    lx = 175
    color = st.LIGHT[light] if light is not None else "none"
    fig.patches.append(
        matplotlib.patches.Ellipse(
            ((lx + 8) * fx, 84 * fy),
            16 * fx,
            16 * fy,
            transform=fig.transFigure,
            facecolor=color,
            edgecolor=st.FAINT if light is None else color,
            zorder=41,
        )
    )
    fig.text(
        (lx + 24) * fx,
        78 * fy,
        st.LIGHT_NAME.get(light, "none"),
        color=st.TEXT,
        fontsize=11,
        fontweight="bold",
        zorder=40,
    )
    fig.text(lx * fx, 56 * fy, "signal on route", color=st.FAINT, fontsize=8.5, zorder=40)
    if f["timing"]:
        times = "  ·  ".join(f"{label} {value:.0f} ms" for label, value in f["timing"][:2])
        fig.text(lx * fx, 32 * fy, times, color=st.MUTED, fontsize=8.5, zorder=40)

    # Right column: legend card.
    has = f["has"]
    items = [("ribbon", st.PLAN, f"{f['plan_label']}, next {horizon:g} s")]
    items += [("line", st.other_color(k), label) for k, (label, _) in enumerate(f["others"])]
    items += [("dash", st.DRIVEN, "path driven next")] * has["odometry"]
    items += [("fill", st.ROUTE, "route")] * has["route"]
    items += [
        ("box", st.CLASS[1][1], "car"),
        ("box", st.CLASS[2][1], "truck / bus"),
        ("box", st.CLASS[7][1], "pedestrian"),
    ] * has["objects"]
    items += [("bar", st.LIGHT["red"], "signal at stop line")] * (has["signals"] and has["map"])
    items += [("dot", "#5cc8ff", "LiDAR, by height")] * has["cloud"]
    # Two columns; a label too long for one takes a whole row.
    cells, col, row_ = [], 0, 0
    for item in items:
        wide = len(item[2]) > 19
        if col == 1 and wide:
            col, row_ = 0, row_ + 1
        cells.append((col, row_))
        col, row_ = (0, row_ + 1) if wide or col == 1 else (1, row_)
    rows = row_ + (col == 1)
    ly = 20 + 300 + 14 + 200 + 14
    row_h = min(32.0, (720 - 20 - ly - 22) / rows)  # rows close up to fit above the bottom
    card(fig, cx, 1 - (ly + 22 + rows * row_h) * fy, cw, (22 + rows * row_h) * fy)
    for (kind, c, label), (col, row_) in zip(items, cells):
        xx = cx + (14 + col * 146) * fx
        yy = 1 - (ly + 26 + row_ * row_h) * fy
        if kind == "dash":
            for d in range(3):
                fig.patches.append(
                    matplotlib.patches.Rectangle(
                        (xx + d * 7 * fx, yy),
                        4 * fx,
                        3 * fy,
                        transform=fig.transFigure,
                        color=c,
                        zorder=41,
                    )
                )
        elif kind == "box":
            fig.patches.append(
                matplotlib.patches.Rectangle(
                    (xx, yy - 4 * fy),
                    18 * fx,
                    11 * fy,
                    transform=fig.transFigure,
                    facecolor=to_rgba(c, 0.25),
                    edgecolor=c,
                    lw=1.1,
                    zorder=41,
                )
            )
        elif kind == "dot":
            for d, cc in enumerate(("#2f4f8f", "#3b82f6", "#5cc8ff", "#e0f6ff")):
                fig.patches.append(
                    matplotlib.patches.Circle(
                        (xx + (2 + d * 5) * fx, yy + 1.5 * fy),
                        2.2 * fx,
                        transform=fig.transFigure,
                        color=cc,
                        zorder=41,
                    )
                )
        else:
            h = {"ribbon": 5, "line": 2.5, "fill": 10, "bar": 4}[kind]
            fig.patches.append(
                matplotlib.patches.Rectangle(
                    (xx, yy + (1.5 - h / 2) * fy),
                    18 * fx,
                    h * fy,
                    transform=fig.transFigure,
                    color=to_rgba(c, 0.45 if kind == "fill" else 1.0),
                    zorder=41,
                )
            )
        fig.text(xx + 26 * fx, yy - 2 * fy, label, color=st.MUTED, fontsize=8.2, zorder=42)


def render_frame(task):
    run_dir, out_dir, index, opts = task
    f = render.get_frame(run_dir, index, opts)
    w = opts.get("width", 1280)
    h = w * 9 // 16
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100 * w / 1280.0, facecolor=st.BG)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_scene(ax, Camera(w, h, opts.get("clean", False)), f, w, h)
    if opts.get("clean"):
        draw_clean_overlay(fig, f, opts)
    else:
        draw_overlay(fig, f, opts)
    path = os.path.join(out_dir, f"{index:05d}.png")
    fig.savefig(path, facecolor=st.BG)
    plt.close(fig)
    return path


def add_chase_args(ap):
    g = ap.add_argument_group("chase camera")
    g.add_argument("--width", type=int, default=1280, help="frame width in pixels; 16:9")
    g.add_argument("--clean", action="store_true", help="only the scene, caption and speed")


def main(argv):
    ap = argparse.ArgumentParser(prog="autoware-viz chase", description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    render.add_selection_args(ap)
    render.add_render_args(ap)
    add_chase_args(ap)
    args = ap.parse_args(argv)
    opts = render.options(args, width=args.width, clean=args.clean)
    run = render.get_run(args.run_dir, opts)
    indices = run.frames_between(args.start, args.end)[:: args.every]
    render.render_all(render_frame, args.run_dir, args.out_dir, indices, opts, args.jobs)
