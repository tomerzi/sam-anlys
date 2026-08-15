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
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
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
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle='round',
                zorder=3)
    # Joints
    for idx, (jx, jy) in enumerate(joints):
        size = joint_size * 1.8 if idx == 0 else joint_size
        ax.scatter(jx, jy, s=size, c=color, zorder=4, alpha=alpha)
    # Head circle
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


def _perp(v, width):
    """Return a unit perpendicular vector scaled by width."""
    n = np.array([-v[1], v[0]], dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([width, 0.0])
    return n / norm * width


def _limb_quad(a, b, w_a, w_b):
    """
    Return the 4 corners of a tapered limb segment from joint a to joint b.
    w_a / w_b are half-widths at each end.
    """
    d = b - a
    pa = _perp(d, w_a)
    pb = _perp(d, w_b)
    return np.array([a + pa, a - pa, b - pb, b + pb])


def draw_mesh(ax, joints, skin_color, outline_color, alpha=0.82, zorder=2):
    """
    Draw a body mesh over the skeleton using filled, tapered polygons for
    each limb segment and the torso, plus edge lines to simulate mesh facets.
    """
    j = joints  # shorthand

    # ── limb segment definitions: (joint_a, joint_b, half_w_a, half_w_b) ──
    segments = [
        # torso (wide trapezoid: shoulders → hips)
        # shoulders centre = midpoint of j[2] and j[5]; hips = j[8],j[11]
        (None, None, None, None, 'torso'),
        # upper arms
        (2, 3, 0.038, 0.028),   # r upper arm
        (5, 6, 0.038, 0.028),   # l upper arm
        # lower arms
        (3, 4, 0.028, 0.018),   # r forearm
        (6, 7, 0.028, 0.018),   # l forearm
        # upper legs
        (8, 9, 0.060, 0.048),   # r thigh
        (11, 12, 0.060, 0.048), # l thigh
        # lower legs
        (9, 10, 0.048, 0.030),  # r shin
        (12, 13, 0.048, 0.030), # l shin
        # neck
        (1, None, None, None, 'neck'),
    ]

    polys = []
    # ── torso ──
    r_sh, l_sh = j[2], j[5]
    r_hp, l_hp = j[8], j[11]
    sh_mid = (r_sh + l_sh) / 2
    hp_mid = (r_hp + l_hp) / 2
    neck = j[1]
    torso_pts = np.array([
        neck + _perp(r_sh - l_sh, 0.05),
        neck - _perp(r_sh - l_sh, 0.05),
        l_hp + _perp(l_sh - r_sh, 0.04),
        r_hp + _perp(r_sh - l_sh, 0.04),
    ])
    polys.append(plt.Polygon(torso_pts, closed=True,
                              facecolor=skin_color, edgecolor=outline_color,
                              lw=0.6, alpha=alpha, zorder=zorder))

    # ── limb quads ──
    simple = [
        (2, 3, 0.040, 0.028),
        (5, 6, 0.040, 0.028),
        (3, 4, 0.028, 0.016),
        (6, 7, 0.028, 0.016),
        (8, 9, 0.062, 0.048),
        (11, 12, 0.062, 0.048),
        (9, 10, 0.048, 0.028),
        (12, 13, 0.048, 0.028),
    ]
    for a_idx, b_idx, wa, wb in simple:
        quad = _limb_quad(j[a_idx], j[b_idx], wa, wb)
        polys.append(plt.Polygon(quad, closed=True,
                                  facecolor=skin_color, edgecolor=outline_color,
                                  lw=0.6, alpha=alpha, zorder=zorder))

    # ── neck ──
    nk_quad = _limb_quad(j[1], j[0], 0.030, 0.022)
    polys.append(plt.Polygon(nk_quad, closed=True,
                              facecolor=skin_color, edgecolor=outline_color,
                              lw=0.6, alpha=alpha, zorder=zorder))

    for p in polys:
        ax.add_patch(p)

    # ── head ──
    head_x, head_y = j[0]
    head_r = 0.060
    head_c = Circle((head_x, head_y + head_r * 0.3), head_r,
                     facecolor=skin_color, edgecolor=outline_color,
                     lw=0.8, alpha=alpha, zorder=zorder + 1)
    ax.add_patch(head_c)

    # ── mesh grid lines inside torso (subtle) ──
    for frac in [0.33, 0.66]:
        mid_top = (j[1] + j[2]) * frac + (j[1] + j[5]) * (1 - frac) * 0.5
    # horizontal lines across torso bands
    for frac in np.linspace(0.2, 0.8, 4):
        p_top = neck * (1 - frac) + hp_mid * frac
        ax.plot(
            [p_top[0] - 0.09, p_top[0] + 0.09],
            [p_top[1], p_top[1]],
            color=outline_color, lw=0.4, alpha=alpha * 0.5, zorder=zorder + 1
        )

    # ── mesh lines on legs ──
    for (a_idx, b_idx) in [(8, 9), (11, 12), (9, 10), (12, 13)]:
        a, b = j[a_idx], j[b_idx]
        for frac in [0.35, 0.65]:
            mid = a * (1 - frac) + b * frac
            ax.plot(
                [mid[0] - 0.025, mid[0] + 0.025],
                [mid[1], mid[1]],
                color=outline_color, lw=0.4, alpha=alpha * 0.5, zorder=zorder + 1
            )


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
