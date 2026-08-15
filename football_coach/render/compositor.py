"""
Compositor: assembles the final side-by-side coaching video.

Output layout (1280 × 720):
┌─────────────────────────────────────────────────┐
│            FOOTBALL MOVEMENT COACH  [score]      │  ← header 56px
├───────────────────────┬─────────────────────────┤
│   ORIGINAL MOVEMENT   │   CORRECTED MOVEMENT    │  ← panels 564px
│   skeleton + errors   │   ideal 3D animation    │
├───────────────────────┴─────────────────────────┤
│  [phase label]  [scrolling coaching tips]        │  ← footer 100px
└─────────────────────────────────────────────────┘

Each panel renders via matplotlib into a numpy image then assembled with
OpenCV into the final mp4.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Circle
from pathlib import Path
from typing import Optional

from analyze.pose_extractor import PoseFrame
from analyze.error_detector import AnalysisResult, DetectedError
from reference.ideal_shot import ideal_pose
from render.body_3d import (
    draw_player, draw_error_ring, draw_annotation,
    setup_panel_ax, draw_grass,
)

# ── output dimensions ─────────────────────────────────────────────────────────

OUT_W, OUT_H = 1280, 720
HEADER_H = 56
FOOTER_H = 100
PANEL_H = OUT_H - HEADER_H - FOOTER_H   # 564
PANEL_W = OUT_W // 2                     # 640

DPI = 96
FPS = 24

JERSEY_BAD  = '#c0392b'   # red — original (bad) player
JERSEY_GOOD = '#1abc9c'   # teal — corrected player

# ── annotation layout: offset relative to joint ───────────────────────────────
_ANN_OFFSETS = {
    'head_down':        (-0.42,  0.08),
    'body_upright':     (-0.44,  0.02),
    'arms_down':        (-0.46,  0.04),
    'plant_foot_close': ( 0.14,  0.08),
    'low_knee_drive':   ( 0.16,  0.04),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _fig_to_bgr(fig) -> np.ndarray:
    """Render matplotlib figure to H×W×3 BGR numpy array."""
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)          # RGBA
    plt.close(fig)
    rgb = img[:, :, :3]                 # drop alpha
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _scale_joints(joints: np.ndarray, from_frame_wh, target_xlim, target_ylim):
    """
    If joints came from MediaPipe (0–1 normalized), just return them.
    They're already in the coordinate space draw_player expects.
    """
    return joints


def _joints_in_panel_space(joints: np.ndarray) -> np.ndarray:
    """Remap joints from [0,1] video space to panel axes space [-.05, 1.15]."""
    return joints  # already in [0,1]; axes xlim/ylim handle the mapping


# ── panel renderers ───────────────────────────────────────────────────────────

def render_left_panel(joints: np.ndarray,
                       orig_frame_bgr: Optional[np.ndarray],
                       active_errors: list[DetectedError],
                       show_ann: bool = True) -> np.ndarray:
    """
    Original panel: grass background (or real frame), 3D skeleton, error rings.
    Returns BGR image (PANEL_H × PANEL_W × 3).
    """
    pw = PANEL_W / DPI
    ph = PANEL_H / DPI
    fig, ax = plt.subplots(figsize=(pw, ph), dpi=DPI)
    fig.patch.set_facecolor('#0d1117')
    setup_panel_ax(ax)

    if orig_frame_bgr is not None:
        # Show real video frame as background
        rgb = cv2.cvtColor(orig_frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (PANEL_W, PANEL_H))
        ax.imshow(rgb_resized, extent=[-0.05, 1.15, -0.05, 2.15],
                  aspect='auto', zorder=0, alpha=0.55)
    else:
        draw_grass(ax)

    draw_player(ax, joints, JERSEY_BAD, alpha=0.92,
                skeleton_alpha=0.5, zorder=2)

    if show_ann:
        for err in active_errors:
            draw_error_ring(ax, joints, err.joint_idx, color=err.color)
            if show_ann:
                off = _ANN_OFFSETS.get(err.error_id, (-0.35, 0.04))
                draw_annotation(ax, joints, err.joint_idx,
                                 err.label_he, err.fix_he, off, err.color)

    # Panel label
    ax.text(0.5, 2.06, 'ORIGINAL', ha='center', fontsize=10,
            fontweight='bold', color='#ff6b6b', transform=ax.transData, zorder=20)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _fig_to_bgr(fig)


def render_right_panel(joints: np.ndarray,
                        ghost_joints: list[np.ndarray] = None) -> np.ndarray:
    """
    Corrected panel: ideal movement 3D animation.
    Returns BGR image (PANEL_H × PANEL_W × 3).
    """
    pw = PANEL_W / DPI
    ph = PANEL_H / DPI
    fig, ax = plt.subplots(figsize=(pw, ph), dpi=DPI)
    fig.patch.set_facecolor('#0d1117')
    setup_panel_ax(ax)
    draw_grass(ax)

    # Motion trail ghosts
    if ghost_joints:
        for i, gj in enumerate(ghost_joints):
            ga = 0.08 + 0.05 * i
            draw_player(ax, gj, JERSEY_GOOD, alpha=ga,
                        skeleton_alpha=0.15, zorder=1)

    draw_player(ax, joints, JERSEY_GOOD, alpha=0.94,
                skeleton_alpha=0.45, zorder=2)

    ax.text(0.5, 2.06, 'CORRECTED', ha='center', fontsize=10,
            fontweight='bold', color='#2ecc71', transform=ax.transData, zorder=20)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _fig_to_bgr(fig)


def render_header(action: str, score: int, frame_frac: float) -> np.ndarray:
    """Top bar: title, technique score meter, phase indicator."""
    fig, ax = plt.subplots(figsize=(OUT_W / DPI, HEADER_H / DPI), dpi=DPI)
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.text(0.5, 0.58, 'FOOTBALL MOVEMENT COACH',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='white', transform=ax.transAxes)
    ax.text(0.5, 0.12,
            f'Action: {action.upper()}',
            ha='center', va='bottom', fontsize=7.5, color='#888',
            transform=ax.transAxes)

    # Score meter (right side)
    score_color = '#2ecc71' if score >= 70 else '#f39c12' if score >= 50 else '#e74c3c'
    ax.text(0.94, 0.55, f'{score}', ha='center', va='center',
            fontsize=15, fontweight='bold', color=score_color,
            transform=ax.transAxes)
    ax.text(0.94, 0.12, 'SCORE', ha='center', fontsize=6,
            color='#666', transform=ax.transAxes)

    # Progress bar
    bar_x = 0.06
    ax.add_patch(plt.Rectangle((bar_x, 0.10), 0.82, 0.14,
                                facecolor='#1c2130', transform=ax.transAxes, zorder=1))
    ax.add_patch(plt.Rectangle((bar_x, 0.10), 0.82 * frame_frac, 0.14,
                                facecolor='#3498db', transform=ax.transAxes, zorder=2))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _fig_to_bgr(fig)


def render_footer(errors: list[DetectedError], scroll_t: float,
                  phase_label: str) -> np.ndarray:
    """Bottom bar: phase name + scrolling coaching tips."""
    fig, ax = plt.subplots(figsize=(OUT_W / DPI, FOOTER_H / DPI), dpi=DPI)
    fig.patch.set_facecolor('#060a14')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # Divider line
    ax.plot([0, 1], [0.92, 0.92], color='#223344', lw=1.5,
            transform=ax.transAxes, clip_on=False)

    ax.text(0.01, 0.72, f'  {phase_label}', ha='left', va='center',
            fontsize=8, color='#3498db', fontweight='bold',
            transform=ax.transAxes)

    # Tips carousel
    tips = []
    for err in errors:
        tips.append(f'[{err.label_he}]  {err.fix_he}')
    if not tips:
        tips = ['תנועה טובה! המשך לתרגל.']

    n = len(tips)
    x_offset = -scroll_t % 1.0
    for i, tip in enumerate(tips):
        x = x_offset + i / n
        if -0.3 < x < 1.1:
            ax.text(x, 0.32, f'  {tip}  ', ha='left', va='center',
                    fontsize=8.5, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a2540',
                              edgecolor='#2c4a7c', alpha=0.85),
                    transform=ax.transAxes)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _fig_to_bgr(fig)


# ── main compose function ─────────────────────────────────────────────────────

def compose_video(
    frames_bgr: list[np.ndarray],      # original video frames
    pose_frames: list[PoseFrame],
    result: AnalysisResult,
    output_path: str | Path,
    ffmpeg_bin: str,
    slow_replay: bool = True,
) -> Path:
    """
    Full pipeline → write coaching mp4.

    Phases:
      0.00–0.18: INTRO   – both sides show stance, no annotations
      0.18–0.55: LIVE    – normal speed playback with error rings appearing
      0.55–0.80: REPLAY  – slow-motion key moment with full annotations
      0.80–1.00: TIPS    – split freeze + big tips
    """
    output_path = Path(output_path)
    import subprocess, os, tempfile

    n_orig = len(pose_frames)
    ideal_frames = _build_ideal_sequence(n_orig, result)

    # Compute total output frames
    n_live   = n_orig
    n_replay = n_orig * 2 if slow_replay else n_orig
    n_tips   = FPS * 3
    n_total  = int(FPS * 1.5) + n_live + n_replay + n_tips  # intro + live + replay + tips

    frame_dir = output_path.parent / '_cmp_frames'
    frame_dir.mkdir(exist_ok=True)

    out_idx = 0

    def _write(canvas: np.ndarray):
        nonlocal out_idx
        p = frame_dir / f'frame_{out_idx:05d}.png'
        cv2.imwrite(str(p), canvas)
        out_idx += 1

    def _assemble(left_bgr, right_bgr, header_bgr, footer_bgr) -> np.ndarray:
        panels = np.hstack([
            cv2.resize(left_bgr,  (PANEL_W, PANEL_H)),
            cv2.resize(right_bgr, (PANEL_W, PANEL_H)),
        ])
        divider = np.full((PANEL_H, 2, 3), 40, dtype=np.uint8)
        panels[:, PANEL_W-1:PANEL_W+1] = divider
        header = cv2.resize(header_bgr, (OUT_W, HEADER_H))
        footer = cv2.resize(footer_bgr, (OUT_W, FOOTER_H))
        return np.vstack([header, panels, footer])

    # ── INTRO (1.5 s) ──────────────────────────────────────────────────────────
    n_intro = int(FPS * 1.5)
    for fi in range(n_intro):
        t_intro = fi / max(1, n_intro - 1)
        orig_f  = frames_bgr[0] if frames_bgr else None
        j_orig  = pose_frames[0].joints
        j_good  = ideal_frames[0]
        alpha   = min(1.0, t_intro * 2)
        left  = render_left_panel(j_orig, orig_f, [], show_ann=False)
        right = render_right_panel(j_good)
        hdr   = render_header(result.action, result.technique_score, 0.0)
        ftr   = render_footer([], 0.0, 'INTRO')
        canvas = _assemble(left, right, hdr, ftr)
        _write(canvas)

    # ── LIVE playback ─────────────────────────────────────────────────────────
    for fi in range(n_orig):
        frame_frac = (n_intro + fi) / max(1, n_total - 1)
        j_orig = pose_frames[fi].joints
        j_good = ideal_frames[fi]

        orig_bgr = frames_bgr[fi] if fi < len(frames_bgr) else None

        # Errors appear gradually through live phase
        visible_t = fi / max(1, n_orig - 1)
        active_errors = [e for e in result.errors
                         if e.frame_range[0] <= fi <= e.frame_range[1]]
        show_ann = visible_t > 0.30

        left  = render_left_panel(j_orig, orig_bgr, active_errors, show_ann=show_ann)
        right = render_right_panel(j_good)

        phase = _phase_label(fi, result)
        scroll = (fi / FPS) * 0.12

        hdr   = render_header(result.action, result.technique_score, frame_frac)
        ftr   = render_footer(active_errors, scroll, phase)
        _write(_assemble(left, right, hdr, ftr))

    # ── SLOW REPLAY around key frame ──────────────────────────────────────────
    kf = result.key_frame
    replay_window = list(range(max(0, kf - n_orig // 3), min(n_orig, kf + n_orig // 3)))
    replay_seq = replay_window * 2 if slow_replay else replay_window

    for ri, fi in enumerate(replay_seq):
        frame_frac = (n_intro + n_orig + ri) / max(1, n_total - 1)
        j_orig = pose_frames[fi].joints
        j_good = ideal_frames[fi]
        orig_bgr = frames_bgr[fi] if fi < len(frames_bgr) else None

        active_errors = result.errors  # all errors visible in replay

        # Ghost trail in right panel
        ghosts = [ideal_frames[max(0, fi - g)] for g in [4, 3, 2, 1]]

        left  = render_left_panel(j_orig, orig_bgr, active_errors, show_ann=True)
        right = render_right_panel(j_good, ghost_joints=ghosts)

        scroll = 0.15 + ri / max(1, len(replay_seq)) * 0.3
        hdr   = render_header(result.action, result.technique_score, frame_frac)
        ftr   = render_footer(result.errors, scroll, 'SLOW REPLAY')
        _write(_assemble(left, right, hdr, ftr))

    # ── TIPS freeze ───────────────────────────────────────────────────────────
    j_orig = pose_frames[kf].joints
    j_good = ideal_frames[kf]
    orig_bgr = frames_bgr[kf] if kf < len(frames_bgr) else None
    for ti in range(n_tips):
        frame_frac = (n_intro + n_live + n_replay + ti) / max(1, n_total - 1)
        scroll = 0.45 + ti / FPS * 0.18

        left  = render_left_panel(j_orig, orig_bgr, result.errors, show_ann=True)
        right = render_right_panel(j_good)
        hdr   = render_header(result.action, result.technique_score, frame_frac)
        ftr   = render_footer(result.errors, scroll, 'COACHING TIPS')
        _write(_assemble(left, right, hdr, ftr))

    # ── ENCODE ────────────────────────────────────────────────────────────────
    input_pattern = str(frame_dir / 'frame_%05d.png')
    cmd = [
        ffmpeg_bin, '-y',
        '-framerate', str(FPS),
        '-i', input_pattern,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '20', '-preset', 'fast',
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Cleanup frames
    for f in frame_dir.glob('frame_*.png'):
        f.unlink()
    frame_dir.rmdir()

    return output_path


# ── internal helpers ──────────────────────────────────────────────────────────

def _build_ideal_sequence(n: int, result: AnalysisResult) -> list[np.ndarray]:
    """Interpolate ideal pose sequence to match n_frames."""
    return [ideal_pose(i / max(1, n - 1)) for i in range(n)]


def _phase_label(fi: int, result: AnalysisResult) -> str:
    for name, (s, e) in result.phases.items():
        if s <= fi < e:
            return {'approach': 'APPROACH', 'contact': 'CONTACT',
                    'followthrough': 'FOLLOW-THROUGH'}.get(name, name.upper())
    return ''
