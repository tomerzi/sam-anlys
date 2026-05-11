import numpy as np
from scipy.stats import skew

# ── Keypoint indices ──────────────────────────────────────────────────
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Left/right pairs for horizontal flip augmentation
LR_PAIRS = [
    (L_EYE, R_EYE), (L_EAR, R_EAR),
    (L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST),
    (L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE),
]


def _crop_width_from_bbox(item: dict) -> float:
    """ViTPose keypoints x,y are in padded-crop pixel space; width matches bbox span."""
    bbox = item.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        return float(bbox[2] - bbox[0])
    k = np.array(item["keypoints"], dtype=np.float32)
    if k.size == 0:
        return 1.0
    # Fallback if old JSON lacks bbox: span of x plus margin (rare)
    return float(np.nanmax(k[:, 0]) - np.nanmin(k[:, 0])) + 1e-3


def horizontal_flip_keypoints(kpts: np.ndarray, crop_width: float) -> np.ndarray:
    """Mirror about vertical midline of the crop, then swap left/right joints."""
    kpts = np.asarray(kpts, dtype=np.float32).copy()
    w = float(crop_width)
    kpts[:, 0] = w - kpts[:, 0]
    for l, r in LR_PAIRS:
        kpts[[l, r]] = kpts[[r, l]]
    return kpts


# ── Geometry helpers ──────────────────────────────────────────────────

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at joint b formed by a-b-c, in degrees [0, 180]."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    Normalize (17, 3) keypoints to be translation- and scale-invariant.
    Origin → hip midpoint. Scale → torso height (hip-to-shoulder distance).
    """
    kpts = kpts.copy()
    hip_mid = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2.0
    shoulder_mid = (kpts[L_SHOULDER, :2] + kpts[R_SHOULDER, :2]) / 2.0
    torso_h = np.linalg.norm(shoulder_mid - hip_mid) + 1e-8
    kpts[:, :2] = (kpts[:, :2] - hip_mid) / torso_h
    return kpts


def frame_to_vector(kpts: np.ndarray) -> np.ndarray:
    """
    Convert (17, 3) keypoints → 1-D feature vector (59 dims):
        34  normalized x,y positions
        17  confidence scores
         8  joint angles (normalized 0-1)
    """
    kpts = normalize_keypoints(kpts)
    xy   = kpts[:, :2].flatten()       # 34
    conf = kpts[:, 2]                   # 17

    p = kpts[:, :2]
    angles = np.array([
        _angle(p[L_SHOULDER], p[L_ELBOW], p[L_WRIST]),    # left elbow
        _angle(p[R_SHOULDER], p[R_ELBOW], p[R_WRIST]),    # right elbow
        _angle(p[L_HIP],      p[L_KNEE],  p[L_ANKLE]),    # left knee
        _angle(p[R_HIP],      p[R_KNEE],  p[R_ANKLE]),    # right knee
        _angle(p[L_SHOULDER], p[L_HIP],   p[L_KNEE]),     # left hip flex
        _angle(p[R_SHOULDER], p[R_HIP],   p[R_KNEE]),     # right hip flex
        _angle(p[L_SHOULDER], p[R_SHOULDER], p[R_HIP]),   # shoulder tilt
        _angle(p[L_HIP],      p[R_HIP],   p[R_KNEE]),     # hip tilt
    ], dtype=np.float32) / 180.0       # → [0, 1]

    return np.concatenate([xy, conf, angles]).astype(np.float32)  # 59


# ── Window feature aggregation ────────────────────────────────────────

def build_window_features(
    keypoints_list: list[dict],
    window_size: int = 30,
    stride: int = 5,
) -> tuple[np.ndarray, list[tuple]]:
    """
    Sliding window over per-frame keypoints (list order = consecutive tracked frames).

    Each window yields aggregated statistics:
        mean, std, min, max, range, skewness  ×  118 (pos + velocity)
    = 708 features per window.

    Returns:
        X             (n_windows, 708)
        frame_ranges  list of (start, end) **indices into keypoints_list** (half-open
                      [start, end)), not necessarily contiguous raw video frame indices
                      if the detector dropped frames during occlusion / re-ID.
    """
    # Build per-frame vectors
    vectors = np.array([
        frame_to_vector(np.array(item["keypoints"]))
        for item in keypoints_list
    ])                                          # (T, 59)

    # First-order velocity (frame diff)
    velocity = np.zeros_like(vectors)
    velocity[1:] = vectors[1:] - vectors[:-1]  # (T, 59)

    combined = np.concatenate([vectors, velocity], axis=1)  # (T, 118)

    T = len(combined)
    windows, ranges = [], []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        w = combined[start:end]                 # (window_size, 118)

        agg = np.concatenate([
            w.mean(axis=0),
            w.std(axis=0),
            w.min(axis=0),
            w.max(axis=0),
            w.max(axis=0) - w.min(axis=0),
            np.apply_along_axis(skew, 0, w),
        ])                                      # 6 × 118 = 708

        windows.append(agg)
        ranges.append((start, end))

    return np.array(windows, dtype=np.float32), ranges


# ── Data augmentation ─────────────────────────────────────────────────

def augment_sequence(kpts_list: list[dict]) -> list[list[dict]]:
    """
    7× augmentation on a keypoint sequence:
        original, horizontal flip,
        speed ×0.8, speed ×1.2,
        3× Gaussian noise
    """
    augmented = [kpts_list]

    # 1. Horizontal flip (crop pixel space: x' = W - x, then swap L/R joints)
    flipped = []
    for item in kpts_list:
        w = _crop_width_from_bbox(item)
        kpts = horizontal_flip_keypoints(np.array(item["keypoints"]), w)
        flipped.append({**item, "keypoints": kpts.tolist()})
    augmented.append(flipped)

    # 2. Speed jitter (temporal resampling)
    n = len(kpts_list)
    for speed in [0.8, 1.2]:
        idx = np.linspace(0, n - 1, max(1, int(n * speed))).astype(int)
        idx = np.clip(idx, 0, n - 1)
        augmented.append([kpts_list[i] for i in idx])

    # 3. Gaussian noise on xy positions
    for sigma in [0.01, 0.02, 0.03]:
        noisy = []
        for item in kpts_list:
            kpts = np.array(item["keypoints"]).copy()
            kpts[:, :2] += np.random.normal(0, sigma, kpts[:, :2].shape)
            noisy.append({**item, "keypoints": kpts.tolist()})
        augmented.append(noisy)

    return augmented   # list of 7 sequences
