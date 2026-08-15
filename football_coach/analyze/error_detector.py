"""
Biomechanical error detection for football shooting technique.

Each rule returns an Error when triggered, with:
  - error_id: stable string key
  - label_he: Hebrew label for annotation
  - fix_he: Hebrew fix instruction
  - severity: 0-1 (1 = worst)
  - frame_range: (start, end) indices where error is visible
  - joint_idx: which skeleton joint to highlight
  - color: annotation color
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .pose_extractor import PoseFrame


@dataclass
class DetectedError:
    error_id: str
    label_he: str
    fix_he: str
    severity: float        # 0..1
    joint_idx: int         # which joint to ring
    frame_range: tuple     # (start_frame, end_frame)
    color: str = '#ff3333'
    score_penalty: float = 0.0


@dataclass
class AnalysisResult:
    action: str                        # 'shot', 'pass', 'dribble', 'unknown'
    errors: list[DetectedError]
    technique_score: int               # 0–100
    key_frame: int                     # frame index of peak action moment
    phases: dict = field(default_factory=dict)  # {phase_name: (start, end)}


# ── helpers ──────────────────────────────────────────────────────────────────

def _angle_deg(a, b, c):
    """Angle at vertex b formed by a-b-c."""
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def _vec_angle_to_vertical(p1, p2):
    """Angle (deg) of segment p1→p2 from vertical (0 = straight up)."""
    d = p2 - p1
    return float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]) + 1e-9)))


def _median_joints(frames: list[PoseFrame]) -> np.ndarray:
    """Median joint position across all frames (robust to outliers)."""
    stacked = np.stack([f.joints for f in frames], axis=0)
    return np.median(stacked, axis=0)


# ── action classification ─────────────────────────────────────────────────────

def classify_action(frames: list[PoseFrame]) -> tuple[str, int]:
    """
    Returns (action_name, key_frame_idx).
    Heuristic: look for max height of kicking ankle relative to hip.
    """
    if not frames:
        return 'unknown', 0

    # Find the frame where one ankle is highest relative to its hip
    # (indicates kicking motion)
    max_lift = -1.0
    key_frame = len(frames) // 2

    for i, f in enumerate(frames):
        j = f.joints
        r_lift = j[10][1] - j[8][1]   # r_ankle y - r_hip y
        l_lift = j[13][1] - j[11][1]  # l_ankle y - l_hip y
        lift = max(r_lift, l_lift)
        if lift > max_lift:
            max_lift = lift
            key_frame = i

    if max_lift > 0.10:
        return 'shot', key_frame
    elif max_lift > 0.04:
        return 'pass', key_frame
    else:
        return 'dribble', key_frame


# ── individual error rules ────────────────────────────────────────────────────

def _check_head_down(frames, key_frame, action):
    """Head too low / looking at feet instead of target."""
    errors = []
    bad_count = 0
    start = max(0, key_frame - 5)
    end = min(len(frames), key_frame + 8)
    check_frames = frames[start:end]

    for f in check_frames:
        j = f.joints
        neck_y = j[1][1]
        head_y = j[0][1]
        # Head should be at least 0.15 above neck; also nose x should be
        # roughly over the neck (not tilted heavily down)
        head_above_neck = head_y - neck_y
        if head_above_neck < 0.09:
            bad_count += 1

    if bad_count >= len(check_frames) * 0.5:
        severity = min(1.0, bad_count / max(1, len(check_frames)))
        errors.append(DetectedError(
            error_id='head_down',
            label_he='ראש למטה',
            fix_he='הרם את הראש — עיניים לכיוון השער',
            severity=severity * 0.85,
            joint_idx=0,
            frame_range=(start, end),
            color='#ff3333',
            score_penalty=12,
        ))
    return errors


def _check_body_upright(frames, key_frame, action):
    """Torso too vertical — not leaning into the ball."""
    errors = []
    j = frames[key_frame].joints
    torso_angle = _vec_angle_to_vertical(j[14], j[1])   # torso_mid→neck
    # Good form: 10-25° forward lean; bad: < 5°
    if torso_angle < 6:
        errors.append(DetectedError(
            error_id='body_upright',
            label_he='גוף ישר מדי',
            fix_he='הטה את הגוף קדימה לכיוון הכדור',
            severity=0.80,
            joint_idx=1,
            frame_range=(max(0, key_frame - 3), min(len(frames), key_frame + 6)),
            color='#ff6600',
            score_penalty=15,
        ))
    return errors


def _check_arms_down(frames, key_frame, action):
    """Arms not used for balance — wrists below hips."""
    errors = []
    j = frames[key_frame].joints
    r_wrist_y = j[4][1]
    l_wrist_y = j[7][1]
    r_hip_y   = j[8][1]
    l_hip_y   = j[11][1]
    hip_y     = (r_hip_y + l_hip_y) / 2

    if r_wrist_y < hip_y and l_wrist_y < hip_y:
        errors.append(DetectedError(
            error_id='arms_down',
            label_he='ידיים נמוכות',
            fix_he='פרוס ידיים לצדדים לשיווי משקל',
            severity=0.65,
            joint_idx=7,
            frame_range=(max(0, key_frame - 2), min(len(frames), key_frame + 5)),
            color='#ffaa00',
            score_penalty=10,
        ))
    return errors


def _check_plant_foot(frames, key_frame, action):
    """Plant foot too close to ball / bad positioning."""
    errors = []
    j = frames[key_frame].joints
    # Determine kicking leg: ankle with higher y is kicking
    if j[10][1] > j[13][1]:
        plant_ankle = j[13]
        kick_ankle  = j[10]
        plant_jidx  = 13
    else:
        plant_ankle = j[10]
        kick_ankle  = j[13]
        plant_jidx  = 10

    # Plant foot should be roughly beside ball (similar x), not far behind
    # We approximate ball position as slightly in front of plant ankle
    lateral_dist = abs(plant_ankle[0] - kick_ankle[0])
    if lateral_dist < 0.06:   # feet too close together horizontally
        errors.append(DetectedError(
            error_id='plant_foot_close',
            label_he='רגל מצע קרובה מדי',
            fix_he='מקם את רגל המצע לצד הכדור, לא מאחוריו',
            severity=0.70,
            joint_idx=plant_jidx,
            frame_range=(max(0, key_frame - 4), min(len(frames), key_frame + 3)),
            color='#ff3399',
            score_penalty=12,
        ))
    return errors


def _check_knee_drive(frames, key_frame, action):
    """Kicking knee not driving through — too low at contact."""
    errors = []
    j = frames[key_frame].joints
    if j[10][1] > j[13][1]:
        kick_knee = j[9];  kick_hip = j[8]
    else:
        kick_knee = j[12]; kick_hip = j[11]

    knee_hip_diff = kick_knee[1] - kick_hip[1]   # positive = knee above hip
    if knee_hip_diff < -0.05:   # knee well below hip at contact
        errors.append(DetectedError(
            error_id='low_knee_drive',
            label_he='ברך נמוכה',
            fix_he='הנע את הברך קדימה ומעלה לפני הנגיחה',
            severity=0.75,
            joint_idx=9 if j[10][1] > j[13][1] else 12,
            frame_range=(max(0, key_frame - 2), min(len(frames), key_frame + 6)),
            color='#aa00ff',
            score_penalty=13,
        ))
    return errors


# ── main analysis function ────────────────────────────────────────────────────

_RULES = [
    _check_head_down,
    _check_body_upright,
    _check_arms_down,
    _check_plant_foot,
    _check_knee_drive,
]

_MAX_PENALTY = sum([12, 15, 10, 12, 13])  # sum of all score_penalty values


def analyse(frames: list[PoseFrame]) -> AnalysisResult:
    """Run all error rules and return a complete AnalysisResult."""
    if not frames:
        return AnalysisResult('unknown', [], 50, 0)

    action, key_frame = classify_action(frames)

    all_errors: list[DetectedError] = []
    for rule in _RULES:
        try:
            all_errors.extend(rule(frames, key_frame, action))
        except Exception:
            pass

    total_penalty = sum(e.score_penalty for e in all_errors)
    score = max(10, round(100 - total_penalty * 100 / _MAX_PENALTY))

    n = len(frames)
    phases = {
        'approach':  (0,              n // 3),
        'contact':   (n // 3,         2 * n // 3),
        'followthrough': (2 * n // 3, n - 1),
    }

    return AnalysisResult(
        action=action,
        errors=all_errors,
        technique_score=score,
        key_frame=key_frame,
        phases=phases,
    )
