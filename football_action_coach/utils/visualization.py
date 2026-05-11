import cv2
import numpy as np

# COCO skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),                    # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),            # arms
    (5, 11), (6, 12), (11, 12),                          # torso
    (11, 13), (13, 15), (12, 14), (14, 16),              # legs
]

JOINT_COLOR   = (0, 0, 255)     # red  — keypoints
BONE_COLOR    = (0, 255, 0)     # green — skeleton
GOOD_COLOR    = (0, 200, 0)
OK_COLOR      = (0, 200, 255)
BAD_COLOR     = (0, 0, 200)


def _headline_color(headline: str) -> tuple[int, int, int]:
    u = headline.upper()
    if u.startswith("BAD"):
        return BAD_COLOR
    if u.startswith("OK"):
        return OK_COLOR
    if u.startswith("GOOD"):
        return GOOD_COLOR
    return (180, 180, 180)


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,       # (17, 3): x, y, confidence
    conf_thresh: float = 0.3,
    radius: int = 5,
) -> np.ndarray:
    img  = image.copy()
    kpts = np.array(keypoints)

    for i, j in SKELETON:
        if kpts[i, 2] > conf_thresh and kpts[j, 2] > conf_thresh:
            cv2.line(
                img,
                tuple(map(int, kpts[i, :2])),
                tuple(map(int, kpts[j, :2])),
                BONE_COLOR, 2, cv2.LINE_AA,
            )

    for x, y, c in kpts:
        if c > conf_thresh:
            cv2.circle(img, (int(x), int(y)), radius, JOINT_COLOR, -1, cv2.LINE_AA)

    return img


def draw_prediction(
    image: np.ndarray,
    headline: str,
    prob: float,
    frame_range: tuple[int, int] | None = None,
) -> np.ndarray:
    img = image.copy()
    color = _headline_color(headline)
    text = f"{headline}  {prob:.0%}"

    cv2.rectangle(img, (0, 0), (380, 72), (0, 0, 0), -1)
    cv2.putText(
        img, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA,
    )

    if frame_range:
        rng = f"window idx {frame_range[0]}–{frame_range[1]}"
        cv2.putText(
            img, rng, (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

    return img


def save_prediction_strip(
    crops: list[dict],
    keypoints_list: list[dict],
    headlines: list[str],
    probs: np.ndarray,
    frame_ranges: list[tuple],
    out_path: str,
    stride: int = 5,
    thumb_w: int = 160,
) -> None:
    """
    Save a horizontal strip of thumbnails, one per window, annotated.
    ``headlines`` = one human-readable grade string per window (e.g. BAD / OK / GOOD).
    """
    kp_map = {item["frame_idx"]: item["keypoints"] for item in keypoints_list}
    thumbs = []

    for (start, end), headline, prob in zip(frame_ranges, headlines, probs):
        mid = (start + end) // 2
        available = [c for c in crops if abs(c["frame_idx"] - mid) < stride * 2]
        if not available:
            continue
        crop = min(available, key=lambda c: abs(c["frame_idx"] - mid))["crop"]

        thumb = cv2.resize(crop, (thumb_w, int(thumb_w * crop.shape[0] / crop.shape[1])))

        if mid in kp_map:
            kpts = np.array(kp_map[mid])
            scale = thumb_w / crop.shape[1]
            kpts[:, :2] *= scale
            thumb = draw_keypoints(thumb, kpts)

        thumb = draw_prediction(thumb, headline, float(prob), (start, end))
        thumbs.append(thumb)

    if thumbs:
        max_h = max(t.shape[0] for t in thumbs)
        padded = [
            cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            for t in thumbs
        ]
        strip = np.hstack(padded)
        cv2.imwrite(out_path, strip)
        print(f"[Viz] Strip saved → {out_path}")
