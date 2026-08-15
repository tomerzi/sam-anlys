#!/usr/bin/env python3
"""
Football Movement Coach — main entry point.

Usage:
    python app.py --video path/to/shot.mp4 --out output/coaching.mp4
    python app.py --demo                     # generate demo input + run

Pipeline (mirrors pipeline.yaml):
    1. extract_poses   — MediaPipe → 15-joint skeleton per frame
    2. classify_action — what action? (shot / pass / dribble)
    3. detect_errors   — biomechanical rule engine → error list + score
    4. compose_video   — side-by-side coaching video

Powered by OpenMontage architecture.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio_ffmpeg


def _banner():
    print('\n' + '='*60)
    print('  FOOTBALL MOVEMENT COACH')
    print('  Powered by OpenMontage pipeline')
    print('='*60)


def run(video_path: Path, output_path: Path):
    import cv2
    from analyze.pose_extractor import PoseExtractor, smooth_joints
    from analyze.error_detector import analyse
    from render.compositor import compose_video

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    print(f'\n[1/4] Extracting poses from: {video_path.name}')
    t0 = time.time()
    extractor = PoseExtractor(model_complexity=1)
    meta = extractor.video_meta(video_path)
    pose_frames = extractor.extract(video_path)
    pose_frames = smooth_joints(pose_frames, window=3)
    print(f'      {len(pose_frames)} frames @ {meta["fps"]:.0f}fps  '
          f'({time.time()-t0:.1f}s)')

    if not pose_frames:
        print('[ERROR] No poses detected. Try a clearer video with a single player.')
        sys.exit(1)

    print(f'\n[2/4] Classifying action...')
    from analyze.error_detector import classify_action
    action, key_frame = classify_action(pose_frames)
    print(f'      Action: {action.upper()}  |  Key frame: {key_frame}')

    print(f'\n[3/4] Detecting technique errors...')
    result = analyse(pose_frames)
    score_color = '\033[92m' if result.technique_score >= 70 else \
                  '\033[93m' if result.technique_score >= 50 else '\033[91m'
    print(f'      Technique score: {score_color}{result.technique_score}/100\033[0m')
    for err in result.errors:
        print(f'      [{err.error_id}] {err.label_he}  '
              f'(severity {err.severity:.0%})  → {err.fix_he}')
    if not result.errors:
        print('      No major errors detected — technique looks solid!')

    print(f'\n[4/4] Compositing coaching video...')
    t1 = time.time()

    # Load original frames for left-panel background
    cap = cv2.VideoCapture(str(video_path))
    frames_bgr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames_bgr.append(f)
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    compose_video(
        frames_bgr=frames_bgr,
        pose_frames=pose_frames,
        result=result,
        output_path=output_path,
        ffmpeg_bin=ffmpeg_bin,
        slow_replay=True,
    )
    print(f'      Done in {time.time()-t1:.1f}s')

    size_kb = output_path.stat().st_size // 1024
    print(f'\n✓  Coaching video → {output_path}  ({size_kb} KB)')
    print('='*60 + '\n')
    return output_path


def main():
    _banner()
    parser = argparse.ArgumentParser(description='Football Movement Coach')
    parser.add_argument('--video', type=Path, help='Input video file')
    parser.add_argument('--out',   type=Path,
                        default=Path('output/coaching_result.mp4'))
    parser.add_argument('--demo',  action='store_true',
                        help='Generate synthetic bad-shot demo and analyse it')
    args = parser.parse_args()

    if args.demo:
        print('\n[DEMO] Generating synthetic bad-shot input video...')
        from demo.generate_input import render_input_video
        demo_input = Path('output/demo_input.mp4')
        demo_input.parent.mkdir(exist_ok=True)
        render_input_video(demo_input)
        args.video = demo_input

    if not args.video:
        parser.print_help()
        sys.exit(1)

    if not args.video.exists():
        print(f'[ERROR] Video not found: {args.video}')
        sys.exit(1)

    run(args.video, args.out)


if __name__ == '__main__':
    main()
