"""
Generates a synthetic "bad shot" input video for the demo.
Produces a realistic bad-form shooting animation (no API needed).
The resulting mp4 is fed into app.py as if it were real footage.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

import imageio_ffmpeg
from render.body_3d import draw_player, setup_panel_ax, draw_grass

FPS = 24
DURATION = 3.0   # seconds
N_FRAMES = int(FPS * DURATION)
W, H, DPI = 640, 720, 96


def bad_shot_joints(t: float) -> np.ndarray:
    """
    Bad shooting technique:
    - Body stays upright (no forward lean)
    - Head drops toward ball
    - Arms hang low
    - Kicking knee barely lifts (toe-punt style)
    - Plant foot behind ball
    """
    p = float(np.clip(t, 0, 1))
    j = np.array([
        [0.50, 1.85],   # 0  head (upright)
        [0.50, 1.65],   # 1  neck
        [0.62, 1.55],   # 2  r_shoulder
        [0.72, 1.35],   # 3  r_elbow
        [0.80, 1.15],   # 4  r_wrist (arms down)
        [0.38, 1.55],   # 5  l_shoulder
        [0.28, 1.35],   # 6  l_elbow
        [0.20, 1.15],   # 7  l_wrist (arms down)
        [0.58, 1.10],   # 8  r_hip
        [0.60, 0.60],   # 9  r_knee (plant)
        [0.62, 0.08],   # 10 r_ankle (plant)
        [0.42, 1.10],   # 11 l_hip
        [0.38, 0.60],   # 12 l_knee
        [0.34, 0.08],   # 13 l_ankle
        [0.50, 1.30],   # 14 torso_mid
    ], dtype=float)

    # Kicking leg barely lifts — toe-punt
    j[12] = [0.38 + p * 0.28, 0.60 + p * 0.18]
    j[13] = [0.34 + p * 0.18, 0.08 + p * 0.42]

    # Head tilts down
    j[0][1] -= p * 0.06

    # Arms droop
    j[4][1] -= p * 0.08
    j[7][1] -= p * 0.08

    return j


def render_input_video(out_path: Path) -> Path:
    frames_dir = out_path.parent / '_inp_frames'
    frames_dir.mkdir(exist_ok=True)

    for i in range(N_FRAMES):
        t = i / max(1, N_FRAMES - 1)

        fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
        fig.patch.set_facecolor('#0d1117')
        setup_panel_ax(ax)
        draw_grass(ax)

        j = bad_shot_joints(t)
        draw_player(ax, j, '#c0392b', alpha=0.92, skeleton_alpha=0.5)

        # Ball
        bx = 0.50 + max(0, t - 0.65) * 0.5
        by = 0.055 + max(0, t - 0.65) * 0.02
        ax.add_patch(plt.Circle((bx, by), 0.052, color='#f4a011',
                                 zorder=8))
        ax.add_patch(plt.Circle((bx, by), 0.052, color='black',
                                 fill=False, lw=1.5, zorder=9))

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        p = frames_dir / f'frame_{i:04d}.png'
        fig.savefig(str(p), dpi=DPI, facecolor='#0d1117')
        plt.close(fig)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, '-y',
        '-framerate', str(FPS),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '22', '-preset', 'fast',
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    for f in frames_dir.glob('frame_*.png'):
        f.unlink()
    frames_dir.rmdir()

    print(f'[demo] Input video: {out_path}  ({N_FRAMES} frames @ {FPS}fps)')
    return out_path


if __name__ == '__main__':
    out = Path(__file__).parent.parent / 'output' / 'demo_input.mp4'
    out.parent.mkdir(exist_ok=True)
    render_input_video(out)
