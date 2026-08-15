"""
Pose extraction: video frames → 15-joint skeleton sequences via MediaPipe.

Joint index map (matches our 3D renderer):
  0  head        1  neck        2  r_shoulder   3  r_elbow    4  r_wrist
  5  l_shoulder  6  l_elbow     7  l_wrist      8  r_hip      9  r_knee
  10 r_ankle     11 l_hip       12 l_knee       13 l_ankle    14 torso_mid
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PoseFrame:
    frame_idx: int
    joints: np.ndarray          # (15, 2) in [0,1] coords, y-up
    confidence: float           # mean landmark visibility score
    raw_landmarks: list = field(default_factory=list)


# MediaPipe landmark → our joint index
# MP: 0=nose 11=l_sh 12=r_sh 13=l_el 14=r_el 15=l_wr 16=r_wr
#     23=l_hp 24=r_hp 25=l_kn 26=r_kn 27=l_an 28=r_an
_MP_TO_OUR = {
    # our_idx: mp_idx or (mp_idx_a, mp_idx_b) for midpoint
    2:  12,   # r_shoulder
    3:  14,   # r_elbow
    4:  16,   # r_wrist
    5:  11,   # l_shoulder
    6:  13,   # l_elbow
    7:  15,   # l_wrist
    8:  24,   # r_hip
    9:  26,   # r_knee
    10: 28,   # r_ankle
    11: 23,   # l_hip
    12: 25,   # l_knee
    13: 27,   # l_ankle
}


def _lm(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, 1.0 - lm.y])   # y-flip to y-up


def _midpoint(*lms):
    return np.mean([_lm(*args) if isinstance(args, tuple) else args
                    for args in lms], axis=0)


def _landmarks_to_joints(landmarks) -> np.ndarray:
    """Convert 33 MediaPipe landmarks to our 15-joint array (y-up [0,1])."""
    j = np.zeros((15, 2), dtype=float)

    for our_idx, mp_idx in _MP_TO_OUR.items():
        j[our_idx] = _lm(landmarks, mp_idx)

    # neck = midpoint of shoulders
    j[1] = (j[2] + j[5]) / 2.0

    # head = nose projected up from neck
    nose = _lm(landmarks, 0)
    neck = j[1]
    j[0] = nose + (nose - neck) * 0.15   # push slightly beyond nose

    # torso_mid = centroid of shoulders + hips
    j[14] = (j[2] + j[5] + j[8] + j[11]) / 4.0

    return j


def _mean_visibility(landmarks, indices) -> float:
    return float(np.mean([landmarks[i].visibility for i in indices]))


class PoseExtractor:
    def __init__(self, model_complexity: int = 1):
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def extract(self, video_path: str | Path) -> list[PoseFrame]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames: list[PoseFrame] = []
        idx = 0

        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)

            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark
                joints = _landmarks_to_joints(lms)
                conf = _mean_visibility(lms, list(_MP_TO_OUR.values()) + [0, 11, 12])
                frames.append(PoseFrame(idx, joints, conf, lms))
            else:
                # No detection: carry forward last known or mark as missing
                if frames:
                    prev = frames[-1]
                    frames.append(PoseFrame(idx, prev.joints.copy(), 0.0))
                # else skip until first detection

            idx += 1

        cap.release()
        self._pose.reset()
        return frames

    def video_meta(self, video_path: str | Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        meta = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return meta


def smooth_joints(frames: list[PoseFrame], window: int = 3) -> list[PoseFrame]:
    """Gaussian-smooth joint positions across time to reduce jitter."""
    if len(frames) < window:
        return frames
    n = len(frames)
    stacked = np.stack([f.joints for f in frames], axis=0)  # (N, 15, 2)
    from scipy.ndimage import uniform_filter1d
    try:
        smoothed = uniform_filter1d(stacked, size=window, axis=0)
    except ImportError:
        # Fallback: simple moving average
        smoothed = stacked.copy()
        hw = window // 2
        for i in range(hw, n - hw):
            smoothed[i] = stacked[i - hw: i + hw + 1].mean(axis=0)

    out = []
    for i, f in enumerate(frames):
        out.append(PoseFrame(f.frame_idx, smoothed[i], f.confidence, f.raw_landmarks))
    return out
