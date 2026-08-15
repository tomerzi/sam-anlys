"""
3D body mesh renderer — Lambertian + Phong specular cylindrical shading.
Standalone module shared between original overlay and corrected animation.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Ellipse

# Joint skeleton connectivity
BONES = [
    (0, 1), (1, 14), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (14, 8), (8, 9), (9, 10),
    (14, 11), (11, 12), (12, 13),
]

# ── shading ───────────────────────────────────────────────────────────────────

_LIGHT = np.array([-0.55, 0.72], dtype=float)
_LIGHT /= np.linalg.norm(_LIGHT)


def _perp(v, w):
    n = np.array([-v[1], v[0]], dtype=float)
    d = np.linalg.norm(n)
    return n / d * w if d > 1e-9 else np.array([w, 0.0])


def _cyl_intensity(u: float) -> float:
    nz = np.sqrt(max(0.0, 1.0 - u * u))
    diffuse = max(0.0, _LIGHT[0] * u + _LIGHT[1] * nz)
    nl = _LIGHT[0] * u + _LIGHT[1] * nz
    spec = max(0.0, 2.0 * nl * nz) ** 28 * 0.50
    return min(1.0, 0.22 + 0.65 * diffuse + spec)


def _cylinder(ax, p1, p2, r1, r2, rgba, n=22, alpha=1.0, z=2):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d = p2 - p1
    br, bg, bb, _ = rgba
    for i in range(n):
        ul = -1.0 + 2.0 * i / n
        ur = -1.0 + 2.0 * (i + 1) / n
        s = _cyl_intensity((ul + ur) * 0.5)
        fc = (np.clip(br * s, 0, 1), np.clip(bg * s, 0, 1), np.clip(bb * s, 0, 1))
        quad = np.array([p1 + _perp(d, r1 * ul), p1 + _perp(d, r1 * ur),
                         p2 + _perp(d, r2 * ur), p2 + _perp(d, r2 * ul)])
        ax.add_patch(plt.Polygon(quad, closed=True, facecolor=fc,
                                  edgecolor='none', alpha=alpha, zorder=z))
    dark = (br * 0.25, bg * 0.25, bb * 0.25)
    for side in (-1, 1):
        ax.plot([p1[0] + _perp(d, r1 * side)[0], p2[0] + _perp(d, r2 * side)[0]],
                [p1[1] + _perp(d, r1 * side)[1], p2[1] + _perp(d, r2 * side)[1]],
                color=dark, lw=0.7, alpha=alpha * 0.9, zorder=z + 0.1,
                solid_capstyle='round')


def _sphere(ax, center, radius, rgba, n=24, alpha=1.0, z=4):
    cx, cy = float(center[0]), float(center[1])
    br, bg, bb, _ = rgba
    for i in range(n):
        ul = -1.0 + 2.0 * i / n
        ur = -1.0 + 2.0 * (i + 1) / n
        um = (ul + ur) * 0.5
        h = np.sqrt(max(0.0, 1.0 - um * um)) * radius
        s = _cyl_intensity(um)
        fc = (np.clip(br * s, 0, 1), np.clip(bg * s, 0, 1), np.clip(bb * s, 0, 1))
        ax.add_patch(plt.Polygon(
            [[cx + radius * ul, cy - h], [cx + radius * ur, cy - h],
             [cx + radius * ur, cy + h], [cx + radius * ul, cy + h]],
            closed=True, facecolor=fc, edgecolor='none', alpha=alpha, zorder=z))
    gx, gy = cx - radius * 0.30, cy + radius * 0.32
    glint = (min(1.0, br + 0.55), min(1.0, bg + 0.55), min(1.0, bb + 0.55))
    ax.add_patch(Circle((gx, gy), radius * 0.24, facecolor=glint,
                         edgecolor='none', alpha=alpha * 0.50, zorder=z + 1))
    ax.add_patch(Circle((cx, cy), radius, facecolor='none',
                         edgecolor=(br * 0.25, bg * 0.25, bb * 0.25),
                         lw=0.8, alpha=alpha, zorder=z + 1))


# ── public draw functions ─────────────────────────────────────────────────────

def draw_player(ax, joints, jersey_color: str, alpha: float = 1.0,
                skeleton_alpha: float = 0.45, zorder: int = 2):
    """
    Draw full 3D player: back limbs → torso → front limbs → head.
    Also overlays a faint skeleton wireframe.
    jersey_color: hex color for the body (jersey + shorts colour).
    """
    j = joints
    rgba = np.array(mcolors.to_rgba(jersey_color))
    rgba_back = rgba * np.array([0.68, 0.68, 0.68, 1.0])

    # shadow
    scx = (j[10][0] + j[13][0]) / 2
    ax.add_patch(Ellipse((scx, j[10][1] - 0.01), 0.20, 0.025,
                          facecolor='black', edgecolor='none',
                          alpha=0.22 * alpha, zorder=1))

    # back limbs
    for (a, b, r1, r2) in [(5,6,.031,.023),(6,7,.023,.014),
                             (11,12,.052,.040),(12,13,.040,.023)]:
        _cylinder(ax, j[a], j[b], r1, r2, rgba_back, alpha=alpha*.88, z=zorder)
    for (ci, r) in [(5,.027),(6,.019),(11,.035),(12,.027)]:
        _sphere(ax, j[ci], r, rgba_back, alpha=alpha*.85, z=zorder)

    # torso
    sh_mid = (j[2] + j[5]) / 2
    hp_mid = (j[8] + j[11]) / 2
    w_sh = np.linalg.norm(j[2] - j[5]) / 2 + 0.042
    w_hp = np.linalg.norm(j[8] - j[11]) / 2 + 0.036
    _cylinder(ax, sh_mid, hp_mid, w_sh, w_hp, rgba, n=28, alpha=alpha, z=zorder+1)

    # neck
    _cylinder(ax, j[1], j[0], 0.021, 0.017, rgba, alpha=alpha, z=zorder+1)

    # front limbs
    for (a, b, r1, r2) in [(2,3,.034,.025),(3,4,.025,.014),
                             (8,9,.056,.043),(9,10,.043,.025)]:
        _cylinder(ax, j[a], j[b], r1, r2, rgba, alpha=alpha, z=zorder+2)
    for (ci, r) in [(2,.027),(3,.021),(8,.037),(9,.033)]:
        _sphere(ax, j[ci], r, rgba, alpha=alpha, z=zorder+2)

    # head (skin tone)
    skin = np.array([0.88, 0.72, 0.56, 1.0])
    _sphere(ax, (j[0][0], j[0][1] + 0.055), 0.058, skin,
            n=28, alpha=alpha, z=zorder+3)

    # skeleton wireframe on top
    sk_color = mcolors.to_rgba(jersey_color)[:3]
    for (i, k) in BONES:
        ax.plot([j[i][0], j[k][0]], [j[i][1], j[k][1]],
                color=sk_color, lw=1.2, alpha=skeleton_alpha * alpha,
                solid_capstyle='round', zorder=zorder + 4)
    for idx in range(15):
        if idx == 0:
            continue
        ax.scatter(j[idx][0], j[idx][1], s=18, c=[sk_color],
                   zorder=zorder + 5, alpha=skeleton_alpha * alpha)


def draw_error_ring(ax, joints, joint_idx: int, color: str = '#ff3333',
                    alpha: float = 0.9, zorder: int = 10):
    """Pulsing highlight ring on a specific joint."""
    jx, jy = joints[joint_idx]
    for r, a in [(0.065, 0.3), (0.050, 0.6), (0.038, alpha)]:
        ax.add_patch(Circle((jx, jy), r, facecolor='none',
                             edgecolor=color, lw=2.0, alpha=a, zorder=zorder))


def draw_annotation(ax, joints, joint_idx: int, label: str, fix: str,
                     offset_xy, color: str, zorder: int = 11):
    """Arrow annotation: error ring → text box with label + fix."""
    jx, jy = joints[joint_idx]
    tx, ty = jx + offset_xy[0], jy + offset_xy[1]
    text = f"{label}\n→ {fix}"
    ax.annotate(
        text, xy=(jx, jy), xytext=(tx, ty),
        fontsize=7.0, color='white', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=color, lw=1.4,
                        connectionstyle='arc3,rad=0.15'),
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#0a0a1a',
                  edgecolor=color, alpha=0.90, linewidth=1.4),
        zorder=zorder,
    )


def setup_panel_ax(ax, xlim=(-.05, 1.15), ylim=(-.05, 2.15),
                   bg='#0d1117'):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(bg)


def draw_grass(ax):
    from matplotlib.patches import FancyBboxPatch
    for i in range(12):
        c = '#1a3d1a' if i % 2 == 0 else '#1e451e'
        ax.add_patch(FancyBboxPatch((-.05, i * 0.18 - 0.05), 1.25, 0.18,
                                     boxstyle='square', facecolor=c,
                                     edgecolor='none', zorder=0))
    ax.axhline(0.04, color='white', lw=1.0, alpha=0.35, zorder=1)
