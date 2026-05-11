from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # --- Paths ---
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    # Inference scratch (masks, keypoints, strip) — under project root; works on Windows/Linux/macOS
    outputs_dir: Path = Path("outputs")

    # --- YOLO ---
    yolo_model: str = "yolo11x.pt"       # or yolov8x.pt
    yolo_conf: float = 0.5
    yolo_iou: float = 0.45

    # --- ByteTrack (via supervision) ---
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    max_missing_frames: int = 30          # re-select UI after N lost frames

    # --- SAM 2 ---
    sam2_checkpoint: str = "checkpoints/sam2_hiera_large.pt"
    sam2_config: str = "sam2_hiera_l.yaml"
    use_sam2: bool = True                 # set False to skip segmentation

    # --- ViTPose (mmpose) ---
    vitpose_config: str = (
        "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/"
        "ViTPose_huge_coco_256x192.py"
    )
    vitpose_checkpoint: str = "checkpoints/vitpose_huge.pth"
    keypoint_conf_threshold: float = 0.3

    # --- Sequence / windows ---
    sequence_length: int = 30            # frames per window
    sequence_stride: int = 5             # hop between windows
    crop_pad: int = 20                   # pixel padding around YOLO bbox

    # --- XGBoost ---
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # --- Device ---
    device: str = "cuda"                 # "cpu" if no GPU
