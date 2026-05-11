import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path
from config import Config


class PlayerDetector:
    """
    YOLOv11 person detection + ByteTrack tracking + mouse-click player selection.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = YOLO(config.yolo_model)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.track_thresh,
            lost_track_buffer=config.track_buffer,
            minimum_matching_threshold=config.match_thresh,
        )
        self.selected_id: int | None = None

    # ------------------------------------------------------------------
    # UI: click to lock onto a player
    # ------------------------------------------------------------------

    def _click_select(self, frame: np.ndarray, detections: sv.Detections) -> int | None:
        display = frame.copy()
        result: dict = {}

        for bbox, tid in zip(detections.xyxy, detections.tracker_id):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display, f"ID {tid}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )
        cv2.putText(
            display, "Click a player to track  |  Q = quit",
            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2,
        )

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for bbox, tid in zip(detections.xyxy, detections.tracker_id):
                    x1, y1, x2, y2 = map(int, bbox)
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        result["id"] = int(tid)
                        print(f"[Detector] Locked onto player ID {tid}")
                        cv2.destroyAllWindows()

        cv2.namedWindow("Select Player")
        cv2.setMouseCallback("Select Player", on_click)

        while "id" not in result:
            cv2.imshow("Select Player", display)
            if cv2.waitKey(16) & 0xFF == ord("q"):
                break

        return result.get("id")

    # ------------------------------------------------------------------
    # Main: process full video, return crop list
    # ------------------------------------------------------------------

    def process_video(self, video_path: str) -> list[dict]:
        """
        Detect + track all people. Let user click to pick one player.
        Returns list of:
            {frame_idx, crop (np.ndarray), bbox [x1,y1,x2,y2], bbox_tight}

        When the selected track is missing (occlusion / ID switch), that video
        frame produces **no** entry: the returned list is only successful
        detections, in ascending ``frame_idx``. Downstream sliding windows use
        **consecutive list indices**, not a dense per-frame timeline.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.selected_id = None
        missing_count = 0
        crops: list[dict] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- Detection ---
            results = self.model(
                frame, classes=[0],               # person only
                conf=self.config.yolo_conf,
                iou=self.config.yolo_iou,
                verbose=False,
            )
            detections = sv.Detections.from_ultralytics(results[0])
            detections = self.tracker.update_with_detections(detections)

            if len(detections) == 0:
                frame_idx += 1
                continue

            # --- Player selection (first frame or after loss) ---
            if self.selected_id is None:
                self.selected_id = self._click_select(frame, detections)
                if self.selected_id is None:
                    break

            # --- Filter for selected player ---
            ids = detections.tracker_id
            if ids is not None and self.selected_id in ids:
                missing_count = 0
                mask = ids == self.selected_id
                x1, y1, x2, y2 = map(int, detections.xyxy[mask][0])

                # Padded crop
                pad = self.config.crop_pad
                h, w = frame.shape[:2]
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(w, x2 + pad)
                y2p = min(h, y2 + pad)

                crops.append({
                    "frame_idx": frame_idx,
                    "crop": frame[y1p:y2p, x1p:x2p].copy(),
                    "bbox": [x1p, y1p, x2p, y2p],          # padded
                    "bbox_tight": [x1, y1, x2, y2],         # YOLO bbox
                })
            else:
                missing_count += 1
                if missing_count > self.config.max_missing_frames:
                    print(f"[Detector] Player lost at frame {frame_idx}. Re-select.")
                    self.selected_id = self._click_select(frame, detections)
                    missing_count = 0

            frame_idx += 1

        cap.release()
        print(f"[Detector] {len(crops)} crops extracted for player {self.selected_id}")
        return crops
