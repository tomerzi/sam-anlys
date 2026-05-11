import json
import numpy as np
from pathlib import Path
from config import Config

# COCO 17 keypoint names — for reference throughout the project
COCO_KEYPOINTS = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]
KP = {name: i for i, name in enumerate(COCO_KEYPOINTS)}


class PoseEstimator:
    """
    ViTPose-H via mmpose.
    Runs top-down pose estimation on masked player crops.
    """

    def __init__(self, config: Config):
        self.config = config
        from mmpose.apis import init_model, inference_topdown

        self._infer = inference_topdown
        self.model = init_model(
            config.vitpose_config,
            config.vitpose_checkpoint,
            device=config.device,
        )

    # ------------------------------------------------------------------

    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Run ViTPose on a single crop (full image = one person).
        Returns (17, 3) array: [x, y, confidence].
        Returns zeros if detection fails.
        """
        import numpy as np

        h, w = image_bgr.shape[:2]
        bboxes = np.array([[0, 0, w, h, 1.0]])   # xyxy + score

        results = self._infer(self.model, image_bgr, bboxes)

        if not results or len(results[0].pred_instances.keypoints) == 0:
            return np.zeros((17, 3), dtype=np.float32)

        kpts = results[0].pred_instances.keypoints[0]         # (17, 2)
        scores = results[0].pred_instances.keypoint_scores[0] # (17,)
        return np.concatenate([kpts, scores[:, None]], axis=1).astype(np.float32)

    # ------------------------------------------------------------------

    def process_crops(self, segmented: list[dict], save_dir: Path) -> list[dict]:
        """
        Extract keypoints for all segmented crops.
        Saves keypoints.json and returns list of dicts.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        all_kp: list[dict] = []

        for item in segmented:
            image = item["masked_crop"]
            kpts = self.estimate(image)

            # Filter low-confidence keypoints to zero
            low_conf = kpts[:, 2] < self.config.keypoint_conf_threshold
            kpts[low_conf] = 0.0

            all_kp.append({
                "frame_idx": item["frame_idx"],
                "keypoints": kpts.tolist(),    # (17, 3)
                "bbox": item["bbox"],
            })

        out_path = save_dir / "keypoints.json"
        with open(out_path, "w") as f:
            json.dump(all_kp, f, indent=2)

        print(f"[PoseEstimator] Keypoints saved → {out_path}")
        return all_kp
