import cv2
import numpy as np
import torch
from pathlib import Path
from config import Config


class PlayerSegmentor:
    """
    SAM 2 offline segmentation.
    Run once per video, save masked crops to disk.
    Set config.use_sam2 = False to skip and use raw crops instead.
    """

    def __init__(self, config: Config):
        self.config = config
        if config.use_sam2:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2 = build_sam2(
                config.sam2_config,
                config.sam2_checkpoint,
                device=config.device,
            )
            self.predictor = SAM2ImagePredictor(sam2)
        else:
            self.predictor = None

    # ------------------------------------------------------------------

    def _segment(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Segment the dominant person in a crop.
        Returns binary mask (H, W), uint8.
        """
        h, w = crop_bgr.shape[:2]
        image_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        with torch.inference_mode(), torch.autocast(
            self.config.device, dtype=torch.bfloat16
        ):
            self.predictor.set_image(image_rgb)

            # Use full crop as bounding box prompt
            box = np.array([[0, 0, w, h]], dtype=np.float32)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )

        # Pick highest-score mask
        best = masks[np.argmax(scores)]
        return best.astype(np.uint8)

    # ------------------------------------------------------------------

    def process_crops(self, crops: list[dict], save_dir: Path) -> list[dict]:
        """
        Segment all crops offline. Saves masked crops to save_dir.
        Returns crops list with added 'masked_crop' key.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for item in crops:
            crop = item["crop"]
            frame_idx = item["frame_idx"]

            if self.predictor is not None:
                mask = self._segment(crop)
                masked = crop.copy()
                masked[mask == 0] = 0           # zero background
            else:
                mask = np.ones(crop.shape[:2], dtype=np.uint8)
                masked = crop.copy()

            # Save to disk
            masked_path = save_dir / f"masked_{frame_idx:06d}.jpg"
            cv2.imwrite(str(masked_path), masked)

            results.append({
                **item,
                "masked_crop": masked,
                "masked_crop_path": str(masked_path),
            })

        print(f"[Segmentor] {len(results)} masked crops saved to {save_dir}")
        return results
