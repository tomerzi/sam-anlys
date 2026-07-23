Football Action Coach

An AI pipeline that analyzes football (soccer) video clips and grades a player's movement quality as Bad / OK / Good using computer vision and machine learning.



How It Works

Video → Detect & Track → Segment (SAM 2) → Pose Estimation (ViTPose) → Features → XGBoost → Grade





Detection & Tracking — YOLO11x detects players; ByteTrack tracks them across frames. You click to lock onto one player.



Segmentation — SAM 2 generates a precise per-frame body mask for the selected player.



Pose Estimation — ViTPose-H extracts 17 COCO body keypoints (joints) per frame.



Feature Engineering — Builds a 59-D per-frame vector (normalized coords + confidence + 8 joint angles), then aggregates 30-frame sliding windows into 708-D feature vectors with 7× augmentation.



Classification — XGBoost classifier trained with Leave-One-Out Cross-Validation (per video) predicts 0=bad, 1=ok, 2=good with per-class probabilities.



Visualization — Outputs a predictions_strip.jpg with skeleton overlays and color-coded grade labels.



Project Structure

football_action_coach/
├── main.py                  # CLI: preprocess / train / infer
├── config.py                # All paths, model params, hyperparameters
├── requirements.txt         # Dependencies
├── pipeline/
│   ├── detector.py          # YOLO + ByteTrack + click-select
│   ├── segmentor.py         # SAM 2 segmentation
│   ├── pose_estimator.py    # ViTPose-H keypoint extraction
│   ├── feature_engineer.py  # Feature vectors, windows, augmentation
│   └── classifier.py        # XGBoost, LOO-CV, save/load
└── utils/
    └── visualization.py     # Skeleton draw, prediction strip

Runtime directories (auto-created):







Path



Contents





data/<video_id>/



Preprocessed clip: meta.json, masks/, keypoints/keypoints.json





models/



xgb_model.pkl, scaler.pkl after training





outputs/infer/<video_stem>/



Inference results: masks, keypoints, predictions_strip.jpg





checkpoints/



SAM 2 + ViTPose weights (not in git — see installation)



Installation

1. Clone & install core dependencies

git clone https://github.com/tomerzi/sam-anlys.git
cd sam-anlys/football_action_coach
pip install -r requirements.txt

2. Install SAM 2

pip install git+https://github.com/facebookresearch/sam2.git
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt -P checkpoints/

3. Install ViTPose (mmpose)

pip install -U openmim
mim install mmengine mmcv mmdet mmpose
wget https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth \
     -O checkpoints/vitpose_huge.pth



GPU required — the pipeline defaults to device: "cuda". Set device: "cpu" in config.py for CPU-only (slow).



Usage

All commands are run from inside the football_action_coach/ directory.

Step 1 — Preprocess labeled videos

Run once per training clip. Labels follow rubric v2: 0=bad, 1=ok, 2=good.

python main.py preprocess --video path/to/clip.mp4 --label 2 --id player_A_good
python main.py preprocess --video path/to/clip2.mp4 --label 0 --id player_B_bad

Step 2 — Train the model

Requires at least 2 preprocessed videos. Runs LOO-CV and fits a final model on all data.

python main.py train

Outputs models/xgb_model.pkl and models/scaler.pkl.

Step 3 — Run inference on a new video

python main.py infer --video path/to/new_clip.mp4

Outputs to outputs/infer/<video_stem>/:





masks/ — per-frame SAM 2 masks



keypoints/ — extracted keypoint JSON



predictions_strip.jpg — visual grade timeline

Example output:

── Predictions ──
  window idx    0– 29 → GOOD (max 87%) [█████████████████░░░] | BAD 5% OK 8% GOOD 87%
  window idx    5– 34 → GOOD (max 91%) [██████████████████░░] | BAD 3% OK 6% GOOD 91%
  window idx   10– 39 → OK   (max 74%) [██████████████░░░░░░] | BAD 12% OK 74% GOOD 14%



Configuration

All settings live in config.py:







Parameter



Default



Description





yolo_model



yolo11x.pt



Detection model





use_sam2



True



Toggle SAM 2 segmentation





sequence_length



30



Frames per classification window





sequence_stride



5



Hop between windows





device



cuda



cuda or cpu





n_estimators



200



XGBoost trees



Grading Rubric







Label



Name



Meaning





0



BAD



Poor technique — major form issues





1



OK



Acceptable — minor corrections needed





2



GOOD



Strong technique — coach-approved form



Label your training clips consistently using this rubric. The model quality depends directly on the consistency of your labels.



Roadmap







Phase



Description



Status





A



Pipeline hardening (cross-platform paths, augmentation fixes)



Done





B



3-level grading (bad/ok/good), LOO-CV metrics, viz



Done





C



Action type field in metadata, ablation studies



In progress





D



Coaching hints — "what to fix" (joint deltas vs good reference)



Planned





E



Packaging — Docker, pinned deps, HTML report



Planned



Tech Stack





Ultralytics YOLO — person detection



Supervision / ByteTrack — multi-object tracking



SAM 2 — player segmentation



ViTPose-H / mmpose — 2D pose estimation



XGBoost + scikit-learn — classification



PyTorch, OpenCV, NumPy

