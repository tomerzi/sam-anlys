"""
Football Player Movement Correction Video
Creates an animated video showing a footballer making a bad shot movement
with skeleton annotations and corrections.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Ellipse
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import imageio_ffmpeg
import subprocess
import os
import tempfile
from pathlib import Path

# ─── Skeleton joint indices ───────────────────────────────────────────────────
# 0=head, 1=neck, 2=r_shoulder, 3=r_elbow, 4=r_wrist,
# 5=l_shoulder, 6=l_elbow, 7=l_wrist,
# 8=r_hip, 9=r_knee, 10=r_ankle,
# 11=l_hip, 12=l_knee, 13=l_ankle,
# 14=torso_mid

BONES = [
    (0, 1),    # head-neck
    (1, 14),   # neck-torso
    (1, 2),    # neck-r_shoulder
    (2, 3),    # r_shoulder-r_elbow
    (3, 4),    # r_elbow-r_wrist
    (1, 5),    # neck-l_shoulder
    (5, 6),    # l_shoulder-l_elbow
    (6, 7),    # l_elbow-l_wrist
    (14, 8),   # torso-r_hip
    (8, 9),    # r_hip-r_knee
    (9, 10),   # r_knee-r_ankle
    (14, 11),  # torso-l_hip
    (11, 12),  # l_hip-l_knee
    (12, 13),  # l_knee-l_ankle
]

# ─── Pose definitions ─────────────────────────────────────────────────────────

def make_bad_shot_pose(t):
    """
    BAD shooting form:
    - Body too upright (no forward lean)
    - Kicking leg rises but knee leads forward (toepunt style)
    - Plant foot too close to ball
    - Arms down / not balanced
    - Head down (looking at feet, not target)
    """
    # Interpolate between neutral and fully extended bad shot
    p = np.clip(t, 0, 1)

    # Base positions (x, y) — right-footed kick
    joints = np.array([
        [0.50, 1.85],   # 0  head
        [0.50, 1.65],   # 1  neck
        [0.62, 1.55],   # 2  r_shoulder
        [0.72, 1.35],   # 3  r_elbow
        [0.80, 1.15],   # 4  r_wrist
        [0.38, 1.55],   # 5  l_shoulder
        [0.28, 1.35],   # 6  l_elbow
        [0.20, 1.15],   # 7  l_wrist
        [0.58, 1.10],   # 8  r_hip
        [0.60, 0.60],   # 9  r_knee
        [0.62, 0.10],   # 10 r_ankle (plant)
        [0.42, 1.10],   # 11 l_hip
        [0.38, 0.60],   # 12 l_knee
        [0.34, 0.10],   # 13 l_ankle
        [0.50, 1.30],   # 14 torso_mid
    ], dtype=float)

    # Animate kicking leg (right = plant, left = kicking in BAD form)
    # BAD: knee-led kick, torso stays vertical, arms down
    kick_knee_x = 0.38 + p * 0.30   # knee swings forward but too high
    kick_knee_y = 0.60 + p * 0.35
    kick_ankle_x = 0.34 + p * 0.20  # toe trails behind knee
    kick_ankle_y = 0.10 + p * 0.60

    joints[12] = [kick_knee_x, kick_knee_y]
    joints[13] = [kick_ankle_x, kick_ankle_y]

    # BAD: torso stays upright — head tilts down toward ball
    head_tilt = p * 0.05
    joints[0][0] += head_tilt   # head leans slightly but wrong way
    joints[0][1] -= 0.0          # no proper lean

    # BAD: arms drop (not used for balance)
    joints[4][1] -= p * 0.10
    joints[7][1] -= p * 0.10

    return joints


def make_good_shot_pose(t):
    """
    GOOD shooting form:
    - Forward body lean into the ball
    - Plant foot beside the ball (not too close/far)
    - Non-kicking knee slightly bent on landing
    - Arms spread wide for balance
    - Head up / eyes toward target
    - Kicking knee comes through first, then ankle snaps
    """
    p = np.clip(t, 0, 1)

    joints = np.array([
        [0.45, 1.85],   # 0  head (lean forward)
        [0.47, 1.65],   # 1  neck
        [0.60, 1.55],   # 2  r_shoulder
        [0.72, 1.40],   # 3  r_elbow
        [0.82, 1.22],   # 4  r_wrist (arms wide)
        [0.34, 1.58],   # 5  l_shoulder
        [0.22, 1.43],   # 6  l_elbow
        [0.10, 1.25],   # 7  l_wrist (arms wide)
        [0.55, 1.08],   # 8  r_hip
        [0.57, 0.58],   # 9  r_knee
        [0.60, 0.08],   # 10 r_ankle (plant — beside ball)
        [0.40, 1.08],   # 11 l_hip
        [0.36, 0.58],   # 12 l_knee
        [0.32, 0.08],   # 13 l_ankle
        [0.47, 1.28],   # 14 torso_mid
    ], dtype=float)

    # Animate kicking (right) leg — good form: follow-through high
    kick_knee_x = 0.40 + p * 0.15
    kick_knee_y = 0.58 + p * 0.30
    kick_ankle_x = 0.32 + p * 0.35  # ankle snaps through
    kick_ankle_y = 0.08 + p * 0.75

    joints[12] = [kick_knee_x, kick_knee_y]
    joints[13] = [kick_ankle_x, kick_ankle_y]

    # GOOD: torso leans forward with kick
    lean = p * 0.08
    joints[0][0] -= lean
    joints[1][0] -= lean * 0.8
    joints[14][0] -= lean * 0.5

    # GOOD: arms spread and rise for balance
    joints[4][1] += p * 0.08
    joints[7][1] += p * 0.08

    return joints


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_skeleton(ax, joints, color, alpha=1.0, lw=3, joint_size=80, head_r=0.04,
                  with_mesh=True, mesh_alpha=None):
    """Draw body mesh then skeleton bones and joint circles."""
    if with_mesh and alpha > 0.3:
        # Pick mesh skin color based on skeleton color (bad=red-tinted, good=green-tinted)
        if color in ('#ef5350',):
            skin = '#c46060'
            outline = '#8b2020'
        elif color in ('#69f0ae',):
            skin = '#4a9e75'
            outline = '#1b6840'
        else:
            skin = '#7a7a9a'
            outline = '#444466'
        ma = (mesh_alpha if mesh_alpha is not None else alpha * 0.75)
        draw_mesh(ax, joints, skin_color=skin, outline_color=outline,
                  alpha=ma, zorder=2)

    for i, j in BONES:
        x = [joints[i][0], joints[j][0]]
        y = [joints[i][1], joints[j][1]]
        ax.plot(x, y, color=color, lw=lw, alpha=alpha * 0.55,
                solid_capstyle='round', zorder=5)
    # Joint dots (skip head=0 — mesh draws it as a sphere)
    for idx, (jx, jy) in enumerate(joints):
        if idx == 0 and with_mesh:
            continue
        size = joint_size
        ax.scatter(jx, jy, s=size, c=color, zorder=6, alpha=alpha * 0.7)
    # Head dot when no mesh
    if not with_mesh:
        head_circle = Circle((joints[0][0], joints[0][1] + head_r),
                              head_r, color=color, fill=True, zorder=4, alpha=alpha)
        ax.add_patch(head_circle)


def draw_ball(ax, pos, color='#f4a011', alpha=1.0):
    """Draw a football."""
    ball = Circle(pos, 0.06, color=color, fill=True, zorder=5, alpha=alpha)
    seam = Circle(pos, 0.06, color='black', fill=False, lw=2, zorder=6, alpha=alpha)
    ax.add_patch(ball)
    ax.add_patch(seam)
    # Simple pentagon pattern suggestion
    for angle in [0, 72, 144, 216, 288]:
        rad = np.radians(angle)
        px = pos[0] + 0.025 * np.cos(rad)
        py = pos[1] + 0.025 * np.sin(rad)
        dot = Circle((px, py), 0.01, color='#222', zorder=7, alpha=alpha)
        ax.add_patch(dot)


def draw_bad_annotations(ax, joints):
    """Overlay red error markers and correction text on bad pose."""
    errors = [
        # (joint_idx, label, text_offset)
        (0,  "[X] ראש למטה\nצריך להרים את הראש!", (-0.38, 0.08)),
        (13, "[X] כף רגל נגררת\nצריך לנגוח עם קצה הרגל!", (0.12, 0.10)),
        (12, "[X] ברך לא מסתובבת\nפתח את הירך!", (0.15, 0.02)),
        (7,  "[X] ידיים למטה\nפרוס ידיים לצדדים!", (-0.42, 0.05)),
        (1,  "[X] גוף ישר מדי\nהטה קדימה לכיוון הכדור!", (-0.42, 0.02)),
    ]
    for j_idx, label, offset in errors:
        jx, jy = joints[j_idx]
        # Red ring
        ring = Circle((jx, jy), 0.055, color='red', fill=False, lw=2.5,
                       zorder=8, alpha=0.9)
        ax.add_patch(ring)
        # Arrow + text
        tx, ty = jx + offset[0], jy + offset[1]
        ax.annotate(
            label,
            xy=(jx, jy),
            xytext=(tx, ty),
            fontsize=7.5,
            color='white',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ff4444', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#cc0000',
                      edgecolor='#ff4444', alpha=0.88),
            zorder=10,
        )


def draw_good_annotations(ax, joints):
    """Overlay green checkmarks and positive feedback on good pose."""
    checks = [
        (0,  "[V] ראש למעלה\nעיניים לכיוון השער!", (-0.40, 0.08)),
        (13, "[V] קצה כף הרגל נוגע\nסיבוב נכון של הקרסול!", (0.12, 0.08)),
        (12, "[V] ברך חוצה קדימה\nפתיחת ירך מלאה!", (0.14, 0.00)),
        (7,  "[V] ידיים פרושות\nשיווי משקל מושלם!", (-0.42, 0.06)),
        (1,  "[V] גוף מוטה קדימה\nמעביר כוח לבעיטה!", (-0.42, 0.00)),
    ]
    for j_idx, label, offset in checks:
        jx, jy = joints[j_idx]
        ring = Circle((jx, jy), 0.055, color='#00e676', fill=False, lw=2.5,
                       zorder=8, alpha=0.9)
        ax.add_patch(ring)
        tx, ty = jx + offset[0], jy + offset[1]
        ax.annotate(
            label,
            xy=(jx, jy),
            xytext=(tx, ty),
            fontsize=7.5,
            color='white',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#00e676', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#00695c',
                      edgecolor='#00e676', alpha=0.88),
            zorder=10,
        )


# ─── 3-D shading helpers ──────────────────────────────────────────────────────

# Light source: upper-left, slightly toward viewer
_LIGHT_2D = np.array([-0.55, 0.72], dtype=float)
_LIGHT_2D /= np.linalg.norm(_LIGHT_2D)


def _perp(v, width):
    """Unit perpendicular to v, scaled by width."""
    n = np.array([-v[1], v[0]], dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([width, 0.0])
    return n / norm * width


def _cyl_intensity(u):
    """
    Lambertian + Phong specular shading for a cylinder at lateral param u ∈ [-1,1].
    Assumes light from upper-left; viewer looking along -Z.
    """
    nz = np.sqrt(max(0.0, 1.0 - u * u))          # surface faces viewer amount
    # 2-D surface normal projected: (u, nz)
    diffuse = max(0.0, _LIGHT_2D[0] * u + _LIGHT_2D[1] * nz)
    ambient = 0.22
    # Specular: reflect L over N, check against view direction (0,0,1)
    #   R = 2(N·L)N - L  →  R_z = 2(N·L)*N_z
    nl = _LIGHT_2D[0] * u + _LIGHT_2D[1] * nz
    spec = max(0.0, 2.0 * nl * nz) ** 28 * 0.50
    return min(1.0, ambient + 0.65 * diffuse + spec)


def _draw_cylinder(ax, p1, p2, r1, r2, rgba, n=22, alpha=1.0, zorder=2):
    """3-D tapered cylinder from p1 (radius r1) to p2 (radius r2)."""
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d = p2 - p1
    br, bg, bb, _ = rgba

    for i in range(n):
        ul = -1.0 + 2.0 * i / n
        ur = -1.0 + 2.0 * (i + 1) / n
        um = (ul + ur) * 0.5

        s = _cyl_intensity(um)
        fc = (np.clip(br * s, 0, 1),
              np.clip(bg * s, 0, 1),
              np.clip(bb * s, 0, 1))

        quad = np.array([
            p1 + _perp(d, r1 * ul),
            p1 + _perp(d, r1 * ur),
            p2 + _perp(d, r2 * ur),
            p2 + _perp(d, r2 * ul),
        ])
        ax.add_patch(plt.Polygon(quad, closed=True, facecolor=fc,
                                  edgecolor='none', alpha=alpha, zorder=zorder))

    # Silhouette edge lines
    dark = (br * 0.25, bg * 0.25, bb * 0.25)
    for side in (-1, 1):
        e1 = p1 + _perp(d, r1 * side)
        e2 = p2 + _perp(d, r2 * side)
        ax.plot([e1[0], e2[0]], [e1[1], e2[1]],
                color=dark, lw=0.7, alpha=alpha * 0.9,
                zorder=zorder + 0.1, solid_capstyle='round')


def _draw_sphere(ax, center, radius, rgba, n=24, alpha=1.0, zorder=4):
    """3-D sphere as a shaded disc with specular highlight."""
    cx, cy = float(center[0]), float(center[1])
    br, bg, bb, _ = rgba

    for i in range(n):
        ul = -1.0 + 2.0 * i / n
        ur = -1.0 + 2.0 * (i + 1) / n
        um = (ul + ur) * 0.5

        h = np.sqrt(max(0.0, 1.0 - um * um)) * radius
        s = _cyl_intensity(um)
        fc = (np.clip(br * s, 0, 1),
              np.clip(bg * s, 0, 1),
              np.clip(bb * s, 0, 1))

        strip = plt.Polygon([
            [cx + radius * ul, cy - h],
            [cx + radius * ur, cy - h],
            [cx + radius * ur, cy + h],
            [cx + radius * ul, cy + h],
        ], closed=True, facecolor=fc, edgecolor='none', alpha=alpha, zorder=zorder)
        ax.add_patch(strip)

    # Specular glint
    gx, gy = cx - radius * 0.30, cy + radius * 0.32
    gr = radius * 0.24
    glint_c = (min(1.0, br + 0.55), min(1.0, bg + 0.55), min(1.0, bb + 0.55))
    ax.add_patch(Circle((gx, gy), gr, facecolor=glint_c, edgecolor='none',
                         alpha=alpha * 0.50, zorder=zorder + 1))
    # Outline
    ax.add_patch(Circle((cx, cy), radius, facecolor='none',
                         edgecolor=(br * 0.25, bg * 0.25, bb * 0.25),
                         lw=0.8, alpha=alpha, zorder=zorder + 1))


def draw_mesh(ax, joints, skin_color, outline_color=None, alpha=0.90, zorder=2):
    """
    Full 3-D body mesh: Lambertian+Specular cylinders for limbs,
    shaded spheres for joints and head, depth-ordered back→front.
    """
    j = joints
    rgba = np.array(mcolors.to_rgba(skin_color))

    # Slightly darker variant for back-facing limbs
    rgba_back = rgba * np.array([0.72, 0.72, 0.72, 1.0])

    # ── ground shadow ─────────────────────────────────────────────────────────
    scx = (j[10][0] + j[13][0]) / 2
    ax.add_patch(Ellipse((scx, 0.035), 0.22, 0.030,
                          facecolor='black', edgecolor='none',
                          alpha=0.28, zorder=1))

    # ── back limbs (left side of player, further from viewer) ─────────────────
    _draw_cylinder(ax, j[5],  j[6],  0.032, 0.024, rgba_back, alpha=alpha * 0.88, zorder=zorder)
    _draw_cylinder(ax, j[6],  j[7],  0.024, 0.014, rgba_back, alpha=alpha * 0.88, zorder=zorder)
    _draw_sphere(ax, j[5],  0.028, rgba_back, alpha=alpha * 0.85, zorder=zorder)
    _draw_sphere(ax, j[6],  0.020, rgba_back, alpha=alpha * 0.85, zorder=zorder)
    _draw_cylinder(ax, j[11], j[12], 0.054, 0.042, rgba_back, alpha=alpha * 0.88, zorder=zorder)
    _draw_cylinder(ax, j[12], j[13], 0.042, 0.024, rgba_back, alpha=alpha * 0.88, zorder=zorder)
    _draw_sphere(ax, j[11], 0.036, rgba_back, alpha=alpha * 0.85, zorder=zorder)
    _draw_sphere(ax, j[12], 0.028, rgba_back, alpha=alpha * 0.85, zorder=zorder)

    # ── torso ─────────────────────────────────────────────────────────────────
    sh_mid = (j[2] + j[5]) / 2
    hp_mid = (j[8] + j[11]) / 2
    w_sh = np.linalg.norm(j[2] - j[5]) / 2 + 0.042
    w_hp = np.linalg.norm(j[8] - j[11]) / 2 + 0.036
    _draw_cylinder(ax, sh_mid, hp_mid, w_sh, w_hp, rgba,
                   n=28, alpha=alpha, zorder=zorder + 1)

    # ── neck ──────────────────────────────────────────────────────────────────
    _draw_cylinder(ax, j[1], j[0], 0.022, 0.018, rgba,
                   alpha=alpha, zorder=zorder + 1)

    # ── front limbs (right side of player) ────────────────────────────────────
    _draw_sphere(ax, j[2],  0.028, rgba, alpha=alpha, zorder=zorder + 2)
    _draw_cylinder(ax, j[2],  j[3],  0.035, 0.026, rgba, alpha=alpha, zorder=zorder + 2)
    _draw_sphere(ax, j[3],  0.022, rgba, alpha=alpha, zorder=zorder + 2)
    _draw_cylinder(ax, j[3],  j[4],  0.026, 0.015, rgba, alpha=alpha, zorder=zorder + 2)

    _draw_sphere(ax, j[8],  0.038, rgba, alpha=alpha, zorder=zorder + 2)
    _draw_cylinder(ax, j[8],  j[9],  0.058, 0.044, rgba, alpha=alpha, zorder=zorder + 2)
    _draw_sphere(ax, j[9],  0.034, rgba, alpha=alpha, zorder=zorder + 2)
    _draw_cylinder(ax, j[9],  j[10], 0.044, 0.026, rgba, alpha=alpha, zorder=zorder + 2)

    # ── head sphere (skin tone) ───────────────────────────────────────────────
    skin_rgba = np.array([0.88, 0.72, 0.56, 1.0])
    _draw_sphere(ax, (j[0][0], j[0][1] + 0.056), 0.060, skin_rgba,
                 n=28, alpha=alpha, zorder=zorder + 3)


def draw_field_background(ax):
    """Draw simple grass field."""
    field = FancyBboxPatch((0, 0), 1, 2, boxstyle='square',
                            facecolor='#2d6a2d', edgecolor='none', zorder=0)
    ax.add_patch(field)
    # Grass stripes
    for i in range(10):
        stripe_color = '#2a622a' if i % 2 == 0 else '#2d6a2d'
        stripe = FancyBboxPatch((0, i * 0.2), 1, 0.2, boxstyle='square',
                                 facecolor=stripe_color, edgecolor='none', zorder=0)
        ax.add_patch(stripe)
    # Ground line
    ax.axhline(0.05, color='white', lw=1.5, alpha=0.5, zorder=1)


# ─── Phase titles ──────────────────────────────────────────────────────────────

PHASE_TITLES = {
    'intro':       ('[ FOOTBALL MOVEMENT ANALYSIS ]', '#ffffff'),
    'bad_static':  ('[X] בעיטה שגויה — BEFORE', '#ff5252'),
    'bad_anim':    ('[X] בעיטה שגויה בתנועה', '#ff5252'),
    'annotate':    ('[?] זיהוי שגיאות', '#ffab40'),
    'good_static': ('[V] בעיטה נכונה — AFTER', '#69f0ae'),
    'good_anim':   ('[V] בעיטה נכונה בתנועה', '#69f0ae'),
    'good_ann':    ('[V] תנועה נכונה מוסברת', '#69f0ae'),
    'outro':       ('[!] סיכום ועצות', '#80d8ff'),
}


# ─── Main render ──────────────────────────────────────────────────────────────

FPS = 24
TOTAL_SECONDS = 18
N_FRAMES = FPS * TOTAL_SECONDS

BALL_POS_BAD  = np.array([0.50, 0.05])   # ball near plant foot (bad — too close)
BALL_POS_GOOD = np.array([0.55, 0.05])   # ball correctly beside plant foot


def compute_phase(frame):
    """Return (phase_name, local_t [0..1]) for a given frame."""
    t = frame / N_FRAMES  # 0..1

    breakpoints = [
        (0.00, 0.06, 'intro'),
        (0.06, 0.17, 'bad_static'),
        (0.17, 0.33, 'bad_anim'),
        (0.33, 0.50, 'annotate'),
        (0.50, 0.60, 'good_static'),
        (0.60, 0.76, 'good_anim'),
        (0.76, 0.90, 'good_ann'),
        (0.90, 1.00, 'outro'),
    ]
    for start, end, name in breakpoints:
        if t < end:
            local_t = (t - start) / (end - start)
            return name, np.clip(local_t, 0, 1)
    return 'outro', 1.0


FRAME_W = 800  # pixels, must be even
FRAME_H = 1000  # pixels, must be even
DPI = 100


def render_frame(frame):
    fig, ax = plt.subplots(figsize=(FRAME_W / DPI, FRAME_H / DPI), dpi=DPI)
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 2.15)
    ax.set_aspect('equal')
    ax.axis('off')

    phase, t = compute_phase(frame)

    # Title bar
    title_text, title_color = PHASE_TITLES.get(phase, ('', 'white'))
    ax.text(0.5, 2.08, title_text, ha='center', va='center',
            fontsize=14, fontweight='bold', color=title_color,
            transform=ax.transData, zorder=20)

    draw_field_background(ax)

    if phase == 'intro':
        # Draw a simple ball graphic
        ball_c = Circle((0.5, 1.5), 0.20, color='#f4a011', zorder=5)
        ball_s = Circle((0.5, 1.5), 0.20, color='black', fill=False, lw=4, zorder=6)
        ax.add_patch(ball_c)
        ax.add_patch(ball_s)
        ax.text(0.5, 0.9, 'ניתוח תנועות שחקן כדורגל', ha='center',
                fontsize=16, fontweight='bold', color='white',
                transform=ax.transData, zorder=5)
        ax.text(0.5, 0.65, 'בעיטה לשער — Before & After',
                ha='center', fontsize=12, color='#aaaaaa',
                transform=ax.transData, zorder=5)

    elif phase == 'bad_static':
        joints = make_bad_shot_pose(0)
        draw_skeleton(ax, joints, color='#ef5350', lw=3.5)
        draw_ball(ax, BALL_POS_BAD)
        # Label
        ax.text(0.5, 0.28, 'שחקן מתכונן לבעיטה — תנוחה שגויה',
                ha='center', fontsize=9, color='#ff8a80',
                transform=ax.transData, zorder=10)

    elif phase == 'bad_anim':
        joints = make_bad_shot_pose(t)
        draw_skeleton(ax, joints, color='#ef5350', lw=3.5)
        # Ball moves slightly on contact
        ball_pos = BALL_POS_BAD.copy()
        if t > 0.6:
            ball_pos[0] += (t - 0.6) * 0.3
            ball_pos[1] += (t - 0.6) * 0.05
        draw_ball(ax, ball_pos)

    elif phase == 'annotate':
        joints = make_bad_shot_pose(0.6)
        draw_skeleton(ax, joints, color='#ef5350', lw=3.5, alpha=0.85)
        draw_ball(ax, BALL_POS_BAD)
        # Annotations fade in
        if t > 0.15:
            draw_bad_annotations(ax, joints)

    elif phase == 'good_static':
        joints = make_good_shot_pose(0)
        draw_skeleton(ax, joints, color='#69f0ae', lw=3.5)
        draw_ball(ax, BALL_POS_GOOD)
        ax.text(0.5, 0.28, 'תנוחת בעיטה נכונה',
                ha='center', fontsize=9, color='#b9f6ca',
                transform=ax.transData, zorder=10)

    elif phase == 'good_anim':
        joints = make_good_shot_pose(t)
        draw_skeleton(ax, joints, color='#69f0ae', lw=3.5)
        ball_pos = BALL_POS_GOOD.copy()
        if t > 0.55:
            ball_pos[0] += (t - 0.55) * 0.50
            ball_pos[1] += (t - 0.55) * 0.20
        draw_ball(ax, ball_pos)
        # Motion trail on kicking foot (no mesh on ghosts)
        if t > 0.2:
            for ti in np.linspace(0, t - 0.1, 5):
                ghost = make_good_shot_pose(ti)
                draw_skeleton(ax, ghost, color='#69f0ae', lw=1,
                               alpha=0.12, joint_size=20, with_mesh=False)

    elif phase == 'good_ann':
        joints = make_good_shot_pose(0.7)
        draw_skeleton(ax, joints, color='#69f0ae', lw=3.5, alpha=0.85)
        draw_ball(ax, BALL_POS_GOOD)
        if t > 0.15:
            draw_good_annotations(ax, joints)

    elif phase == 'outro':
        tips = [
            '[V] הטי את גופך קדימה בזמן הבעיטה',
            '[V] ראש למעלה — עיניים לכיוון השער',
            '[V] פרוס ידיים לצדדים לשיווי משקל',
            '[V] נגח עם קצה הרגל הפנימי/חיצוני',
            '[V] ברך חוצה את הכדור בעת המכה',
            '[V] רגל המצע — לצד הכדור, לא מתחתיו',
        ]
        ax.text(0.5, 1.85, '[!] עצות שיפור — בעיטה לשער',
                ha='center', fontsize=13, fontweight='bold',
                color='#80d8ff', transform=ax.transData, zorder=10)
        n_show = max(1, int(t * len(tips) * 1.3))
        for i, tip in enumerate(tips[:min(n_show, len(tips))]):
            ax.text(0.08, 1.60 - i * 0.22, tip,
                    ha='left', fontsize=10, color='white',
                    transform=ax.transData, zorder=10,
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='#1a237e', alpha=0.7))

    # Frame counter (small, bottom right)
    ax.text(1.10, 0.01, f'{frame+1}/{N_FRAMES}',
            ha='right', fontsize=6, color='#555555',
            transform=ax.transData, zorder=20)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def main():
    output_dir = Path(__file__).parent
    output_path = output_dir / 'football_movement_correction.mp4'
    frames_dir = output_dir / '_frames'
    frames_dir.mkdir(exist_ok=True)

    print(f"Rendering {N_FRAMES} frames at {FPS} fps ...")

    frame_paths = []
    for i in range(N_FRAMES):
        if i % 24 == 0:
            print(f"  frame {i}/{N_FRAMES}")
        fig = render_frame(i)
        p = frames_dir / f'frame_{i:04d}.png'
        fig.savefig(str(p), dpi=DPI,
                    facecolor='#111111', format='png')
        plt.close(fig)
        frame_paths.append(str(p))

    print("Encoding video with ffmpeg ...")
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    input_pattern = str(frames_dir / 'frame_%04d.png')
    cmd = [
        ffmpeg_bin,
        '-y',
        '-framerate', str(FPS),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '20',
        '-preset', 'fast',
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:", result.stderr[-1000:])
        raise RuntimeError("ffmpeg failed")

    print(f"Done! Video saved to: {output_path}")

    # Cleanup frames
    for p in frame_paths:
        os.remove(p)
    frames_dir.rmdir()

    return str(output_path)


if __name__ == '__main__':
    main()
