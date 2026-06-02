from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
import tqdm

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(
    r"D:\HP\OneDrive\Desktop\学校\课程\专业课\大数据综合工程设计\crack-seg\算法\runs\segment\train\weights\best.pt"
)
VIDEO_PATH = Path(
    r"D:\HP\OneDrive\Desktop\学校\课程\专业课\大数据综合工程设计\crack-seg\算法\dataset\videoplayback.mp4"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "tracking_outputs"
CONF_THRESHOLD = 0.05
TRACKER_CFG = "botsort.yaml"


@dataclass
class TrackMemoryItem:
    track_id: int
    class_id: int
    class_name: str
    first_frame: int
    last_frame: int
    first_time: float
    last_time: float
    frame_count: int = 0
    best_conf: float = 0.0
    best_score: float = float("-inf")
    best_bbox: list[float] | None = None
    best_crop_path: str = ""
    best_frame_index: int = -1


def _sanitize_name(name: str) -> str:
    return (
        "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip(
            "_"
        )
        or "unknown"
    )


def _to_int_bbox(box_xyxy: np.ndarray, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return [x1, y1, x2, y2]


def _crop_from_bbox(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2].copy()


def _compute_quality_score(
    image: np.ndarray,
    bbox: list[int],
    confidence: float,
    mask_area_ratio: float,
) -> float:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area_ratio = (bbox_w * bbox_h) / float(max(1, width * height))

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    image_center_x = width / 2.0
    image_center_y = height / 2.0
    max_dist = math.hypot(image_center_x, image_center_y) or 1.0
    center_dist = math.hypot(center_x - image_center_x, center_y - image_center_y)
    center_score = 1.0 - min(1.0, center_dist / max_dist)

    crop = _crop_from_bbox(image, bbox)
    blur_score = 0.0
    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = min(1.0, lap_var / 500.0)

    return (
        0.50 * float(confidence)
        + 0.20 * min(1.0, bbox_area_ratio * 6.0)
        + 0.15 * min(1.0, mask_area_ratio * 1.5)
        + 0.10 * center_score
        + 0.05 * blur_score
    )


def _save_best_crop(
    output_dir: Path,
    memory: TrackMemoryItem,
    image: np.ndarray,
    bbox: list[int],
    frame_index: int,
) -> str:
    crop = _crop_from_bbox(image, bbox)
    class_slug = _sanitize_name(memory.class_name)
    file_name = f"track_{memory.track_id:04d}_{class_slug}_best.jpg"
    crop_path = output_dir / file_name
    if crop.size > 0:
        cv2.imwrite(str(crop_path), crop)

    memory.best_crop_path = str(crop_path)
    memory.best_bbox = [int(v) for v in bbox]
    memory.best_frame_index = frame_index
    return str(crop_path)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"视频文件不存在: {VIDEO_PATH}")

    run_dir = OUTPUT_ROOT / VIDEO_PATH.stem
    crops_dir = run_dir / "best_crops"
    run_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    fps = 25.0
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.release()

    track_memory: dict[int, TrackMemoryItem] = {}
    frame_stats: list[dict] = []
    annotated_writer = None
    track_colors: dict[int, tuple[int, int, int]] = {}

    results = model.track(
        source=str(VIDEO_PATH),
        stream=True,
        persist=True,
        conf=CONF_THRESHOLD,
        tracker=TRACKER_CFG,
        show=False,
        verbose=False,
    )

    for frame_index, result in enumerate(tqdm.tqdm(results, desc="Frames", unit="frame")):
        image = result.orig_img
        if image is None:
            continue

        # initialize annotated video writer on first valid frame
        if annotated_writer is None:
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            annotated_path = run_dir / "annotated.mp4"
            annotated_writer = cv2.VideoWriter(str(annotated_path), fourcc, float(fps), (w, h))

        boxes = result.boxes
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        names = result.names if hasattr(result, "names") else {}

        frame_track_ids: list[int] = []
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            frame_stats.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(frame_index / fps, 3),
                    "detections": 0,
                    "track_ids": [],
                }
            )
            continue

        track_ids = boxes.id.cpu().numpy().astype(int)
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confs = (
            boxes.conf.cpu().numpy()
            if boxes.conf is not None
            else np.ones(len(track_ids))
        )
        classes = (
            boxes.cls.cpu().numpy().astype(int)
            if boxes.cls is not None
            else np.zeros(len(track_ids), dtype=int)
        )

        for idx, track_id in enumerate(track_ids):
            bbox = _to_int_bbox(boxes_xyxy[idx], image.shape[1], image.shape[0])
            class_id = int(classes[idx])
            class_name = str(names.get(class_id, class_id))
            confidence = float(confs[idx])

            mask_area_ratio = 0.0
            if masks is not None and idx < len(masks):
                mask = masks[idx]
                bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                mask_area_ratio = float(mask.sum()) / float(bbox_area)

            score = _compute_quality_score(image, bbox, confidence, mask_area_ratio)
            timestamp_sec = round(frame_index / fps, 3)

            if track_id not in track_memory:
                track_memory[track_id] = TrackMemoryItem(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    first_time=timestamp_sec,
                    last_time=timestamp_sec,
                    frame_count=1,
                    best_conf=confidence,
                    best_score=score,
                )
                _save_best_crop(
                    crops_dir, track_memory[track_id], image, bbox, frame_index
                )
            else:
                memory = track_memory[track_id]
                memory.last_frame = frame_index
                memory.last_time = timestamp_sec
                memory.frame_count += 1
                if confidence > memory.best_conf:
                    memory.best_conf = confidence
                if score > memory.best_score:
                    memory.best_score = score
                    memory.class_id = class_id
                    memory.class_name = class_name
                    _save_best_crop(crops_dir, memory, image, bbox, frame_index)

            frame_track_ids.append(int(track_id))

        # draw annotations on frame and write to annotated video
        try:
            vis = image.copy()
            for idx, track_id in enumerate(track_ids):
                bbox = _to_int_bbox(boxes_xyxy[idx], vis.shape[1], vis.shape[0])
                class_id = int(classes[idx])
                class_name = str(names.get(class_id, class_id))
                color = track_colors.get(int(track_id))
                if color is None:
                    c = np.random.RandomState(int(track_id) & 0xFFFFFFFF).randint(0, 255, size=3)
                    color = (int(c[0]), int(c[1]), int(c[2]))
                    track_colors[int(track_id)] = color
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}:{int(track_id)}"
                txt_pos = (x1, max(0, y1 - 6))
                cv2.putText(vis, label, txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if annotated_writer is not None:
                annotated_writer.write(vis)
        except Exception:
            pass

        frame_stats.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 3),
                "detections": len(track_ids),
                "track_ids": frame_track_ids,
            }
        )

    summary = {
        "video_path": str(VIDEO_PATH),
        "model_path": str(MODEL_PATH),
        "tracker": TRACKER_CFG,
        "conf_threshold": CONF_THRESHOLD,
        "unique_instances": len(track_memory),
        "tracks": [_to_jsonable(asdict(item)) for item in track_memory.values()],
        "frames": _to_jsonable(frame_stats),
    }

    (run_dir / "track_memory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # release writer
    if annotated_writer is not None:
        annotated_writer.release()

    print(f"视频处理完成: {VIDEO_PATH.name}")
    print(f"唯一实例数: {len(track_memory)}")
    print(f"最佳代表图目录: {crops_dir}")


if __name__ == "__main__":
    main()
