"""
Reference ideal shooting pose sequence.
Returns joints (15, 2) in [0,1] y-up space for t ∈ [0,1].
"""
import numpy as np


def ideal_pose(t: float) -> np.ndarray:
    """
    Ideal shot technique at phase t (0=approach stance, 1=full follow-through).
    Based on biomechanics principles:
      - 10-20° forward body lean
      - Non-kicking knee slightly flexed
      - Arms spread wide (shoulder height) for balance
      - Head stays up
      - Kicking knee drives through before ankle snaps
      - Plant foot beside ball (not behind it)
    """
    p = float(np.clip(t, 0, 1))

    j = np.array([
        [0.44, 1.86],   # 0  head (slightly forward)
        [0.46, 1.66],   # 1  neck
        [0.59, 1.56],   # 2  r_shoulder
        [0.71, 1.41],   # 3  r_elbow
        [0.84, 1.24],   # 4  r_wrist (arm wide)
        [0.33, 1.59],   # 5  l_shoulder
        [0.21, 1.44],   # 6  l_elbow
        [0.09, 1.26],   # 7  l_wrist (arm wide)
        [0.55, 1.09],   # 8  r_hip
        [0.57, 0.59],   # 9  r_knee (plant)
        [0.60, 0.08],   # 10 r_ankle (plant — beside ball)
        [0.40, 1.09],   # 11 l_hip
        [0.36, 0.59],   # 12 l_knee
        [0.32, 0.08],   # 13 l_ankle
        [0.47, 1.29],   # 14 torso_mid
    ], dtype=float)

    # ── kicking leg (left): knee drives first, ankle follows ──
    kn_x = 0.36 + p * 0.22
    kn_y = 0.59 + p * 0.36         # knee comes well up
    an_x = 0.32 + p * 0.42
    an_y = 0.08 + p * 0.82         # ankle snaps through to high finish

    j[12] = [kn_x, kn_y]
    j[13] = [an_x, an_y]

    # ── torso leans forward with kick ──
    lean = p * 0.10
    j[0][0]  -= lean
    j[1][0]  -= lean * 0.8
    j[14][0] -= lean * 0.4

    # ── arms rise slightly for balance ──
    j[4][1] += p * 0.10
    j[7][1] += p * 0.10

    return j


def ideal_sequence(n_frames: int) -> list[np.ndarray]:
    """Return a list of n_frames joint arrays covering the full shot cycle."""
    return [ideal_pose(i / max(1, n_frames - 1)) for i in range(n_frames)]
