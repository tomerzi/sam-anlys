"""
Football Action Coach — main pipeline
======================================

Usage
-----
# 1. Preprocess each labeled video (run once per video)
python main.py preprocess --video path/to/video.mp4 --label 2 --id player_A_good

# 2. Train + LOO-CV once all videos are preprocessed
python main.py train

# 3. Run inference on a new video
python main.py infer --video path/to/new_video.mp4

Labels (rubric v2): 0 = bad, 1 = ok, 2 = good. Legacy meta without rubric_version uses 0/1 as bad/good.

Smoke test (from football_action_coach/, after checkpoints + deps)
-------------------------------------------------------------------
1. Put at least two preprocessed folders under data/<video_id>/ (each with
   meta.json + keypoints/keypoints.json). Use preprocess --label 0|1|2
   (0=bad, 1=ok, 2=good) — new meta includes rubric_version=2.
2. python main.py train  → writes models/xgb_model.pkl and scaler.pkl.
3. python main.py infer --video path/to_clip.mp4  → artifacts under
   outputs/infer/<video_stem>/ (masks/, keypoints/, predictions_strip.jpg).

See PROJECT_PLAN.md for folder layout, checkpoint URLs, and phase roadmap.
"""

import argparse
import json
import numpy as np
from pathlib import Path

from config import Config
from pipeline.detector import PlayerDetector
from pipeline.segmentor import PlayerSegmentor
from pipeline.pose_estimator import PoseEstimator
from pipeline.feature_engineer import build_window_features, augment_sequence
from pipeline.classifier import ActionClassifier, rubric_label_name
from utils.visualization import save_prediction_strip


def label_from_meta(meta: dict) -> int:
    """Map stored meta label to canonical rubric: 0=bad, 1=ok, 2=good."""
    lab = int(meta["label"])
    if int(meta.get("rubric_version", 1)) >= 2:
        return lab
    return 2 if lab == 1 else 0


# ── Step 1: Preprocess one video ──────────────────────────────────────

def preprocess(video_path: str, label: int, video_id: str, config: Config):
    print(f"\n{'='*50}")
    print(f"Preprocessing  {video_id}  (label={label})")
    print(f"{'='*50}")

    video_dir = config.data_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # --- Detection + tracking + click selection ---
    detector = PlayerDetector(config)
    crops = detector.process_video(video_path)

    # --- SAM 2 segmentation (offline) ---
    segmentor = PlayerSegmentor(config)
    segmented = segmentor.process_crops(crops, video_dir / "masks")

    # --- ViTPose keypoint extraction ---
    pose = PoseEstimator(config)
    keypoints = pose.process_crops(segmented, video_dir / "keypoints")

    # --- Save metadata (rubric v2: 0=bad, 1=ok, 2=good) ---
    meta = {
        "video_id": video_id,
        "video_path": video_path,
        "label": label,
        "rubric_version": 2,
        "n_frames": len(keypoints),
    }
    with open(video_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Done] {len(keypoints)} frames saved → {video_dir}")
    return keypoints


# ── Step 2: Train ─────────────────────────────────────────────────────

def train(config: Config):
    print(f"\n{'='*50}")
    print("Training  (Leave-One-Out CV)")
    print(f"{'='*50}")

    video_datasets: list[dict] = []

    for video_dir in sorted(config.data_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        meta_path = video_dir / "meta.json"
        kp_path   = video_dir / "keypoints" / "keypoints.json"

        if not meta_path.exists() or not kp_path.exists():
            print(f"  Skipping {video_dir.name} — missing files")
            continue

        with open(meta_path)  as f: meta      = json.load(f)
        with open(kp_path)    as f: keypoints = json.load(f)

        label_canon = label_from_meta(meta)
        video_id = meta["video_id"]

        # Augment: 7× sequences
        aug_seqs = augment_sequence(keypoints)

        X_parts, y_parts = [], []
        for seq in aug_seqs:
            X_win, _ = build_window_features(
                seq,
                window_size=config.sequence_length,
                stride=config.sequence_stride,
            )
            if len(X_win) > 0:
                X_parts.append(X_win)
                y_parts.append(np.full(len(X_win), label_canon, dtype=int))

        if not X_parts:
            print(f"  Skipping {video_id} — no windows generated")
            continue

        X = np.concatenate(X_parts)
        y = np.concatenate(y_parts)

        video_datasets.append({"video_id": video_id, "X": X, "y": y})
        nm = rubric_label_name(label_canon, np.arange(3))
        print(f"  {video_id}: {len(X)} windows  label={label_canon} ({nm})")

    if len(video_datasets) < 2:
        raise ValueError("Need at least 2 preprocessed videos to train.")

    # LOO-CV
    classifier = ActionClassifier(config)
    results = classifier.leave_one_out_cv(video_datasets)

    # Final model on all data
    X_all = np.concatenate([v["X"] for v in video_datasets])
    y_all = np.concatenate([v["y"] for v in video_datasets])
    classifier.fit(X_all, y_all)
    classifier.save(config.models_dir)

    return results


# ── Step 3: Inference on a new video ──────────────────────────────────

def infer(video_path: str, config: Config):
    print(f"\n{'='*50}")
    print(f"Inference  {video_path}")
    print(f"{'='*50}")

    video_stem = Path(video_path).stem
    work_dir = (config.outputs_dir / "infer" / video_stem).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Infer] Scratch directory → {work_dir}")

    # Detect + segment + pose
    detector  = PlayerDetector(config)
    crops     = detector.process_video(video_path)

    segmentor = PlayerSegmentor(config)
    segmented = segmentor.process_crops(crops, work_dir / "masks")

    pose      = PoseEstimator(config)
    keypoints = pose.process_crops(segmented, work_dir / "keypoints")

    # Build features
    X, frame_ranges = build_window_features(
        keypoints,
        window_size=config.sequence_length,
        stride=config.sequence_stride,
    )

    if len(X) == 0:
        print("[Infer] Not enough frames to classify.")
        return

    # Load model and predict
    classifier = ActionClassifier(config)
    classifier.load(config.models_dir)
    labels, proba_max, proba = classifier.predict(X)
    cr = classifier.classes_rubric_

    # Print results (window indices = positions in the tracked keypoint sequence)
    print("\n── Predictions ──")
    for (start, end), lbl, pmax, prow in zip(frame_ranges, labels, proba_max, proba):
        name = rubric_label_name(int(lbl), cr)
        parts = [
            f"{rubric_label_name(int(c), cr)} {prow[j]:.0%}"
            for j, c in enumerate(cr)
        ]
        dist = "  ".join(parts)
        bar = "█" * int(pmax * 20) + "░" * (20 - int(pmax * 20))
        print(f"  window idx {start:4d}–{end:4d}  → {name}  (max {pmax:.0%})  [{bar}]  |  {dist}")

    # Save visual strip
    strip_path = str(work_dir / "predictions_strip.jpg")
    headlines = [rubric_label_name(int(l), cr) for l in labels]
    save_prediction_strip(
        crops, keypoints, headlines, proba_max, frame_ranges, strip_path
    )

    return labels, proba_max, proba, frame_ranges


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Football Action Coach")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess", help="Run detection + pose on a labeled video")
    p.add_argument("--video",  required=True, help="Path to video file")
    p.add_argument(
        "--label",
        required=True,
        type=int,
        choices=[0, 1, 2],
        help="Rubric v2: 0=bad, 1=ok, 2=good",
    )
    p.add_argument("--id",     required=True, dest="video_id",
                   help="Unique name for this video (e.g. player_A_good)")

    sub.add_parser("train", help="Train XGBoost with LOO-CV on all preprocessed videos")

    i = sub.add_parser("infer", help="Run inference on a new video")
    i.add_argument("--video", required=True)

    args   = parser.parse_args()
    config = Config()

    if args.cmd == "preprocess":
        preprocess(args.video, args.label, args.video_id, config)
    elif args.cmd == "train":
        train(config)
    elif args.cmd == "infer":
        infer(args.video, config)


if __name__ == "__main__":
    main()
