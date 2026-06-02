"""
推理模块 - YOLOv8 分割模型封装
支持图像推理、视频逐帧推理、批量推理
"""

import time
import cv2
import numpy as np
import math
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import threading

from app.cause_analyzer import get_cause_analyzer

# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class Detection:
    """单个缺陷检测结果"""
    det_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list          # [x1, y1, x2, y2]  像素坐标
    bbox_norm: list     # [x1, y1, x2, y2]  归一化坐标 0-1
    center: list        # [cx, cy]
    width: int
    height: int
    area_px: int        # 边界框面积 (px²)
    mask_area_px: int   # 掩码面积 (px²)，无掩码时等于 bbox area
    has_mask: bool
    mask_polygon: list  # 轮廓点列表 [[x,y], ...]，用于报告
    mask_rle: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    source_indices: list = field(default_factory=list)
    cause_analysis: dict = field(default_factory=dict)
    track_id: Optional[int] = None
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    first_time: Optional[float] = None
    last_time: Optional[float] = None
    frame_count: Optional[int] = None
    best_frame_index: Optional[int] = None
    best_crop_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameResult:
    """单帧/单图推理结果"""
    frame_index: int            # 视频帧序号；单图时为 0
    timestamp_sec: float        # 视频时间戳；单图时为 0
    image_shape: list           # [H, W, C]
    detections: list            # List[Detection]
    inference_time_ms: float
    total_defects: int = 0
    by_class: dict = field(default_factory=dict)  # {class_name: count}

    def __post_init__(self):
        self.total_defects = len(self.detections)
        for d in self.detections:
            self.by_class[d.class_name] = self.by_class.get(d.class_name, 0) + 1

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "image_shape": self.image_shape,
            "detections": [d.to_dict() for d in self.detections],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "total_defects": self.total_defects,
            "by_class": self.by_class,
        }


# ──────────────────────────────────────────────
# 类别配置
# ──────────────────────────────────────────────

# 颜色表：BGR 格式，每个类别一种颜色（最多支持 20 类）
CLASS_COLORS_BGR = [
    (0,   255, 0  ),   # 0: 绿色
    (0,   0,   255),   # 1: 红色
    (255, 165, 0  ),   # 2: 橙色
    (255, 0,   255),   # 3: 品红
    (0,   255, 255),   # 4: 青色
    (128, 0,   255),   # 5: 紫色
    (255, 255, 0  ),   # 6: 黄色
    (0,   128, 255),   # 7: 橙蓝
    (0,   255, 128),   # 8: 春绿
    (128, 255, 0  ),   # 9: 黄绿
]

MERGE_IOU_THRESHOLD = 0.3


# ──────────────────────────────────────────────
# 后处理与特征提取
# ──────────────────────────────────────────────

def box_iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def resize_mask_to_image(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape != image_shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask.astype(np.uint8)


def merge_instances_by_iou(result, iou_threshold: float = MERGE_IOU_THRESHOLD) -> list[dict]:
    """按 bbox IoU 合并同一裂缝的多个 YOLO 分割实例。"""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy().astype(float)
    confs = (
        result.boxes.conf.cpu().numpy().astype(float)
        if result.boxes.conf is not None
        else np.ones(len(boxes), dtype=float)
    )
    classes = (
        result.boxes.cls.cpu().numpy().astype(int)
        if result.boxes.cls is not None
        else np.zeros(len(boxes), dtype=int)
    )
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    if masks is None:
        return []

    names = result.names if hasattr(result, "names") else {}
    img_h, img_w = result.orig_img.shape[:2]

    parent = list(range(len(boxes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iou(boxes[i], boxes[j]) >= iou_threshold:
                union(i, j)

    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)

    merged_instances = []
    for idxs in groups.values():
        best_idx = max(idxs, key=lambda idx: confs[idx])
        class_id = int(classes[best_idx])
        class_name = names.get(class_id, str(class_id))
        merged_box = np.array([np.inf, np.inf, -np.inf, -np.inf], dtype=float)
        merged_mask = np.zeros((img_h, img_w), dtype=bool)
        merged_conf = float(np.max(confs[idxs]))

        for idx in idxs:
            b = boxes[idx]
            merged_box[0] = min(merged_box[0], b[0])
            merged_box[1] = min(merged_box[1], b[1])
            merged_box[2] = max(merged_box[2], b[2])
            merged_box[3] = max(merged_box[3], b[3])
            mask_full = resize_mask_to_image(masks[idx], (img_h, img_w)) > 0
            merged_mask |= mask_full

        merged_instances.append({
            "bbox": merged_box,
            "mask": merged_mask.astype(np.uint8),
            "indices": [int(idx) for idx in idxs],
            "conf": merged_conf,
            "class_id": class_id,
            "class_name": class_name,
        })

    merged_instances.sort(
        key=lambda item: (item["class_id"], -item["conf"], item["bbox"][0], item["bbox"][1])
    )

    class_counts = {}
    for item in merged_instances:
        class_counts[item["class_id"]] = class_counts.get(item["class_id"], 0) + 1
        item["name"] = f'{item["class_name"]}{class_counts[item["class_id"]]}'

    return merged_instances


def mask_to_polygon(mask: np.ndarray) -> list:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2).tolist()


def encode_mask_rle(mask: np.ndarray, bbox: list[int]) -> dict:
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {}

    crop = np.ascontiguousarray((mask[y1:y2, x1:x2] > 0).astype(np.uint8))
    flat = crop.ravel(order="C")
    if flat.size == 0:
        return {}

    counts = []
    current = int(flat[0])
    run_len = 1
    for value in flat[1:]:
        value = int(value)
        if value == current:
            run_len += 1
        else:
            counts.append(run_len)
            current = value
            run_len = 1
    counts.append(run_len)

    return {
        "origin": [x1, y1],
        "size": [int(crop.shape[0]), int(crop.shape[1])],
        "start": int(flat[0]),
        "counts": counts,
    }


def extract_crack_features(image: np.ndarray, mask: np.ndarray, bbox: list[int]) -> dict:
    x1, y1, x2, y2 = bbox
    bbox_w, bbox_h = max(0, x2 - x1), max(0, y2 - y1)
    bbox_area = bbox_w * bbox_h
    mask_uint8 = mask.astype(np.uint8)
    mask_area = int(mask_uint8.sum())
    rectangularity = mask_area / bbox_area if bbox_area > 0 else 0

    contours, _ = cv2.findContours(
        (mask_uint8 * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    slenderness = 0.0
    major_direction = 0.0
    boundary_complexity = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        rw, rh = rect[1]
        if min(rw, rh) > 0:
            slenderness = max(rw, rh) / min(rw, rh)
        major_direction = rect[2]
        if rect[1][0] < rect[1][1]:
            major_direction = 90 + major_direction
        perimeter = cv2.arcLength(largest, True)
        boundary_complexity = (perimeter ** 2) / mask_area if mask_area > 0 else 0.0

    skeleton_length = 0
    branch_points = 0
    end_points = 0
    try:
        from skimage.morphology import skeletonize

        skel = skeletonize(mask_uint8 > 0).astype(np.uint8)
        skeleton_length = int(skel.sum())
        h, w = skel.shape
        for row in range(1, h - 1):
            for col in range(1, w - 1):
                if skel[row, col] == 1:
                    neighbors = int(skel[row - 1:row + 2, col - 1:col + 2].sum()) - 1
                    if neighbors >= 3:
                        branch_points += 1
                    elif neighbors == 1:
                        end_points += 1
    except Exception:
        pass

    mask_bool = mask_uint8.astype(bool)
    pixels = image[mask_bool] if mask_bool.shape == image.shape[:2] else np.empty((0, 3))
    if len(pixels):
        color_mean = [round(float(v), 2) for v in pixels.mean(axis=0)]
        color_std = [round(float(v), 2) for v in pixels.std(axis=0)]
    else:
        color_mean = [0.0, 0.0, 0.0]
        color_std = [0.0, 0.0, 0.0]

    return {
        "mask_area": mask_area,
        "rectangularity": round(float(rectangularity), 4),
        "slenderness": round(float(slenderness), 4),
        "major_direction": round(float(major_direction), 2),
        "boundary_complexity": round(float(boundary_complexity), 4),
        "skeleton_length": skeleton_length,
        "branch_points": branch_points,
        "end_points": end_points,
        "color_mean_bgr": color_mean,
        "color_std_bgr": color_std,
    }


def crop_from_bbox(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    H, W = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2].copy()


def compute_track_quality(
    image: np.ndarray,
    bbox: list[int],
    confidence: float,
    mask_area_ratio: float,
) -> float:
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area_ratio = (bbox_w * bbox_h) / float(max(1, W * H))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    max_dist = math.hypot(W / 2.0, H / 2.0) or 1.0
    center_dist = math.hypot(cx - W / 2.0, cy - H / 2.0)
    center_score = 1.0 - min(1.0, center_dist / max_dist)

    crop = crop_from_bbox(image, bbox)
    sharpness_score = 0.0
    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, lap_var / 500.0)

    return (
        0.50 * float(confidence)
        + 0.20 * min(1.0, bbox_area_ratio * 6.0)
        + 0.15 * min(1.0, mask_area_ratio * 1.5)
        + 0.10 * center_score
        + 0.05 * sharpness_score
    )


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    H, W = image.shape[:2]
    mask_full = resize_mask_to_image(mask, (H, W)) > 0
    if not mask_full.any():
        return image
    overlay = image.copy()
    overlay[mask_full] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def sanitize_name(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return clean.strip("_") or "unknown"


# ──────────────────────────────────────────────
# 模型封装
# ──────────────────────────────────────────────

class DefectDetector:
    """
    基础设施外观缺陷检测器（YOLOv8-Seg）

    使用方法::

        detector = DefectDetector("./pt/best.pt")
        result = detector.detect_image(cv2_img)
    """

    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str = "./pt/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float  = 0.45,
        device: str = "",          # "" = 自动选择 (CUDA > CPU)
        imgsz: int = 640,
    ):
        self.model_path    = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.imgsz         = imgsz
        self.device        = device
        self.model         = None
        self.class_names: dict[int, str] = {}
        self._load_model()

    # ── 模型加载 ─────────────────────────────

    def _load_model(self):
        """延迟导入 ultralytics，仅在此函数中依赖"""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                f"ultralytics 导入失败: {e}\n"
                f"请在 detection 环境中运行: pip install ultralytics"
            ) from e

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.model = YOLO(str(self.model_path))

        # 提取类别名映射
        self.class_names = self.model.names  # {int: str}
        print(f"[Detector] 模型加载成功: {self.model_path.name}")
        print(f"[Detector] 类别: {self.class_names}")
        print(f"[Detector] 设备: {self.device or 'auto'}")

    # ── 单图推理 ─────────────────────────────

    def detect_image(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float]  = None,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
        analyze_causes: bool = True,
    ) -> FrameResult:
        """
        对单张 BGR numpy 图像进行推理。

        :param image: BGR numpy array (H, W, 3)
        :param conf: 置信度阈值（覆盖初始化参数）
        :param iou:  IOU 阈值（覆盖初始化参数）
        :return: FrameResult
        """
        conf = conf if conf is not None else self.conf_threshold
        iou  = iou  if iou  is not None else self.iou_threshold

        t0 = time.perf_counter()
        with self._lock:
            results = self.model.predict(
                source    = image,
                conf      = conf,
                iou       = iou,
                imgsz     = self.imgsz,
                device    = self.device,
                verbose   = False,
                retina_masks = True,   # 全分辨率掩码
            )
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        detections = self._parse_results(results[0], image)

        frame_result = FrameResult(
            frame_index    = frame_index,
            timestamp_sec  = timestamp_sec,
            image_shape    = list(image.shape),
            detections     = detections,
            inference_time_ms = inference_ms,
        )
        if analyze_causes:
            self.add_cause_analysis(image, frame_result)
        return frame_result

    # ── 批量图像推理 ─────────────────────────

    def detect_batch(
        self,
        images: list[np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float]  = None,
    ) -> list[FrameResult]:
        """对多张图像批量推理，返回 List[FrameResult]"""
        return [
            self.detect_image(img, conf=conf, iou=iou, frame_index=i)
            for i, img in enumerate(images)
        ]

    # ── 视频推理 ─────────────────────────────

    def detect_video(
        self,
        video_path: str,
        conf: Optional[float] = None,
        iou: Optional[float]  = None,
        sample_interval: int  = 1,     # 每隔 N 帧采样一次
        max_frames: int       = 500,   # 最大处理帧数
        progress_callback=None,        # callback(current, total)
    ) -> list[FrameResult]:
        """
        逐帧处理视频文件，返回每帧的 FrameResult 列表。

        :param sample_interval: 采样间隔，1=每帧，5=每5帧
        :param max_frames:      最大处理帧数
        :param progress_callback: 进度回调 fn(processed, total)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = min(max_frames, (total_frames + sample_interval - 1) // sample_interval)

        results: list[FrameResult] = []
        frame_idx  = 0
        processed  = 0

        try:
            while cap.isOpened() and processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    ts = frame_idx / fps
                    result = self.detect_image(
                        frame,
                        conf=conf,
                        iou=iou,
                        frame_index=frame_idx,
                        timestamp_sec=round(ts, 3),
                    )
                    results.append(result)
                    processed += 1

                    if progress_callback:
                        progress_callback(processed, sample_count)

                frame_idx += 1
        finally:
            cap.release()

        return results

    def detect_video_tracking(
        self,
        video_path: str,
        output_dir: str | Path,
        asset_url_prefix: str = "",
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        sample_interval: int = 5,
        max_frames: int = 500,
        tracker: str = "botsort.yaml",
        progress_callback=None,
    ) -> dict:
        output_dir = Path(output_dir)
        crops_dir = output_dir / "crops"
        output_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        sample_interval = max(1, int(sample_interval or 1))
        max_frames = max(1, int(max_frames or 1))
        sample_count = min(max_frames, (total_frames + sample_interval - 1) // sample_interval) if total_frames else max_frames

        annotated_path = output_dir / "annotated.mp4"
        writer = None
        if frame_w > 0 and frame_h > 0:
            writer = cv2.VideoWriter(
                str(annotated_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (frame_w, frame_h),
            )

        conf = self.conf_threshold if conf is None else conf
        iou = self.iou_threshold if iou is None else iou
        track_memory: dict[int, dict] = {}
        timeline: list[dict] = []
        track_colors: dict[int, tuple[int, int, int]] = {}
        frame_idx = 0
        sampled = 0
        started = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(annotated_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(fps),
                        (w, h),
                    )

                annotated = frame.copy()
                frame_track_ids: list[int] = []
                frame_detections = 0

                if sampled < max_frames and frame_idx % sample_interval == 0:
                    sampled += 1
                    track_result = self.model.track(
                        source=frame,
                        persist=True,
                        conf=conf,
                        iou=iou,
                        imgsz=self.imgsz,
                        device=self.device,
                        tracker=tracker,
                        verbose=False,
                        retina_masks=True,
                    )[0]

                    boxes = track_result.boxes
                    masks = track_result.masks.data.cpu().numpy() if track_result.masks is not None else None
                    names = track_result.names if hasattr(track_result, "names") else self.class_names

                    if boxes is not None and len(boxes) > 0:
                        boxes_xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(boxes_xyxy))
                        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(boxes_xyxy), dtype=int)
                        if boxes.id is not None:
                            track_ids = boxes.id.cpu().numpy().astype(int)
                        else:
                            track_ids = np.array([frame_idx * 10000 + i for i in range(len(boxes_xyxy))], dtype=int)

                        frame_detections = len(track_ids)
                        for det_idx, track_id in enumerate(track_ids):
                            H, W = frame.shape[:2]
                            x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[det_idx].tolist()]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(W, x2), min(H, y2)
                            bbox = [x1, y1, x2, y2]
                            class_id = int(classes[det_idx])
                            class_name = str(names.get(class_id, class_id))
                            confidence = float(confs[det_idx])
                            timestamp_sec = round(frame_idx / fps, 3)

                            mask_full = np.zeros((H, W), dtype=np.uint8)
                            mask_area_ratio = 0.0
                            if masks is not None and det_idx < len(masks):
                                mask_full = resize_mask_to_image(masks[det_idx], (H, W))
                                bbox_area = max(1, (x2 - x1) * (y2 - y1))
                                mask_area_ratio = float(mask_full.sum()) / float(bbox_area)

                            score = compute_track_quality(frame, bbox, confidence, mask_area_ratio)
                            memory = track_memory.get(int(track_id))
                            if memory is None:
                                memory = {
                                    "track_id": int(track_id),
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "first_frame": frame_idx,
                                    "last_frame": frame_idx,
                                    "first_time": timestamp_sec,
                                    "last_time": timestamp_sec,
                                    "frame_count": 1,
                                    "best_conf": confidence,
                                    "best_score": score,
                                }
                                track_memory[int(track_id)] = memory
                            else:
                                memory["last_frame"] = frame_idx
                                memory["last_time"] = timestamp_sec
                                memory["frame_count"] += 1
                                memory["best_conf"] = max(float(memory["best_conf"]), confidence)

                            if score >= float(memory.get("best_score", -1e9)):
                                crop = crop_from_bbox(frame, bbox)
                                crop_name = f"track_{int(track_id):04d}_{sanitize_name(class_name)}_best.jpg"
                                crop_path = crops_dir / crop_name
                                if crop.size > 0:
                                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
                                features = extract_crack_features(frame, mask_full, bbox)
                                features["instance_name"] = f"{class_name}-T{int(track_id)}"
                                memory.update({
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "best_score": score,
                                    "best_conf": confidence,
                                    "best_bbox": bbox,
                                    "best_crop_bgr": crop,
                                    "best_crop_path": str(crop_path),
                                    "best_crop_url": f"{asset_url_prefix}/crops/{crop_name}" if asset_url_prefix else "",
                                    "best_frame_index": frame_idx,
                                    "best_time": timestamp_sec,
                                    "best_mask_area": int(mask_full.sum()),
                                    "best_mask_polygon": mask_to_polygon(mask_full),
                                    "best_mask_rle": encode_mask_rle(mask_full, bbox),
                                    "best_features": features,
                                    "image_shape": [H, W, 3],
                                })

                            color = track_colors.get(int(track_id))
                            if color is None:
                                random_color = np.random.RandomState(int(track_id) & 0xFFFFFFFF).randint(0, 255, size=3)
                                color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
                                track_colors[int(track_id)] = color
                            if masks is not None and det_idx < len(masks):
                                annotated = overlay_mask(annotated, masks[det_idx], color, alpha=0.35)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                annotated,
                                f"{class_name}:T{int(track_id)}",
                                (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )
                            frame_track_ids.append(int(track_id))

                    timeline.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 3),
                        "detections": frame_detections,
                        "track_ids": frame_track_ids,
                    })
                    if progress_callback:
                        progress_callback(sampled, sample_count)

                if writer is not None:
                    writer.write(annotated)
                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        detections: list[Detection] = []
        analysis_items = []
        analysis_detections = []
        sorted_tracks = sorted(track_memory.values(), key=lambda item: (str(item.get("class_name", "")), int(item["track_id"])))
        max_h = frame_h
        max_w = frame_w

        for det_id, memory in enumerate(sorted_tracks, start=1):
            bbox = [int(v) for v in memory.get("best_bbox", [0, 0, 0, 0])]
            x1, y1, x2, y2 = bbox
            bw, bh = max(0, x2 - x1), max(0, y2 - y1)
            shape = memory.get("image_shape") or [max_h, max_w, 3]
            H = int(shape[0] or max_h or 1)
            W = int(shape[1] or max_w or 1)
            det = Detection(
                det_id=det_id,
                class_id=int(memory.get("class_id", 0)),
                class_name=str(memory.get("class_name", "")),
                confidence=round(float(memory.get("best_conf", 0.0)), 4),
                bbox=bbox,
                bbox_norm=[
                    round(x1 / W, 4), round(y1 / H, 4),
                    round(x2 / W, 4), round(y2 / H, 4),
                ],
                center=[x1 + bw // 2, y1 + bh // 2],
                width=bw,
                height=bh,
                area_px=bw * bh,
                mask_area_px=int(memory.get("best_mask_area", bw * bh)),
                has_mask=bool(memory.get("best_mask_rle") or memory.get("best_mask_polygon")),
                mask_polygon=memory.get("best_mask_polygon", []),
                mask_rle=memory.get("best_mask_rle", {}),
                features=memory.get("best_features", {}),
                source_indices=[int(memory["track_id"])],
                cause_analysis={"status": "pending"},
                track_id=int(memory["track_id"]),
                first_frame=int(memory["first_frame"]),
                last_frame=int(memory["last_frame"]),
                first_time=float(memory["first_time"]),
                last_time=float(memory["last_time"]),
                frame_count=int(memory["frame_count"]),
                best_frame_index=int(memory.get("best_frame_index", memory["first_frame"])),
                best_crop_url=str(memory.get("best_crop_url", "")),
            )
            crop = memory.get("best_crop_bgr")
            if isinstance(crop, np.ndarray) and crop.size > 0:
                analysis_items.append({"crop_bgr": crop, "class_name": det.class_name})
                analysis_detections.append(det)
            else:
                det.cause_analysis = {"status": "error", "error": "best crop is empty"}
            detections.append(det)

        if analysis_items:
            analyses = get_cause_analyzer().analyze_batch(analysis_items)
            for det, analysis in zip(analysis_detections, analyses):
                det.cause_analysis = analysis

        processing_ms = (time.time() - started) * 1000
        frame_result = FrameResult(
            frame_index=0,
            timestamp_sec=0.0,
            image_shape=[max_h, max_w, 3],
            detections=detections,
            inference_time_ms=processing_ms,
        )

        by_class = {}
        for det in detections:
            by_class[det.class_name] = by_class.get(det.class_name, 0) + 1

        return {
            "results": [frame_result],
            "timeline": timeline,
            "annotated_video_path": str(annotated_path),
            "annotated_video_url": f"{asset_url_prefix}/annotated.mp4" if asset_url_prefix else "",
            "summary": {
                "total_defects": len(detections),
                "unique_instances": len(detections),
                "defective_frames": len([item for item in timeline if item["detections"] > 0]),
                "by_class": by_class,
                "avg_inference_ms": round(processing_ms / max(sampled, 1), 2),
                "processing_time_ms": round(processing_ms, 2),
                "processed_frames": frame_idx,
                "sampled_frames": sampled,
                "sample_interval": sample_interval,
                "tracker": tracker,
            },
        }

    # ── 结果解析 ─────────────────────────────

    def _parse_results(self, result, image: np.ndarray) -> list[Detection]:
        """将 ultralytics Result 对象解析为合并后的 Detection 列表"""
        detections: list[Detection] = []
        H, W = image.shape[:2]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        merged_instances = merge_instances_by_iou(result, MERGE_IOU_THRESHOLD)
        for i, item in enumerate(merged_instances):
            cls_id = int(item["class_id"])
            conf = float(item["conf"])
            cls_name = str(item["class_name"])

            x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            bw, bh = max(0, x2 - x1), max(0, y2 - y1)
            cx, cy = x1 + bw // 2, y1 + bh // 2
            bbox_area = bw * bh

            bbox_norm = [
                round(x1 / W, 4), round(y1 / H, 4),
                round(x2 / W, 4), round(y2 / H, 4),
            ]

            mask_bin = resize_mask_to_image(item["mask"], (H, W))
            mask_area = int(mask_bin.sum())
            mask_poly = mask_to_polygon(mask_bin)
            mask_rle = encode_mask_rle(mask_bin, [x1, y1, x2, y2])
            features = extract_crack_features(image, mask_bin, [x1, y1, x2, y2])
            features["instance_name"] = item.get("name", f"{cls_name}{i + 1}")

            detections.append(Detection(
                det_id      = i + 1,
                class_id    = cls_id,
                class_name  = cls_name,
                confidence  = round(conf, 4),
                bbox        = [x1, y1, x2, y2],
                bbox_norm   = bbox_norm,
                center      = [cx, cy],
                width       = bw,
                height      = bh,
                area_px     = bbox_area,
                mask_area_px = mask_area,
                has_mask    = True,
                mask_polygon = mask_poly,
                mask_rle    = mask_rle,
                features    = features,
                source_indices = item.get("indices", []),
                cause_analysis = {"status": "pending"},
            ))

        # 按置信度降序排列
        detections.sort(key=lambda d: d.confidence, reverse=True)
        # 重新分配 ID
        for idx, d in enumerate(detections):
            d.det_id = idx + 1

        return detections

    def add_cause_analysis(self, image: np.ndarray, frame_result: FrameResult) -> FrameResult:
        """为已有检测结果补充 CLIP/SigLIP 成因分析，不重新执行 YOLO。"""
        analyzer = get_cause_analyzer()
        H, W = image.shape[:2]
        analysis_items = []
        valid_detections = []
        for det in frame_result.detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            crop = image[y1:y2, x1:x2]
            analysis_items.append({
                "crop_bgr": crop,
                "class_name": det.class_name,
            })
            valid_detections.append(det)

        analyses = analyzer.analyze_batch(analysis_items)
        for det, analysis in zip(valid_detections, analyses):
            det.cause_analysis = analysis
        return frame_result

    # ── 工具方法 ─────────────────────────────

    def get_class_color(self, class_id: int) -> tuple[int, int, int]:
        """返回指定类别的 BGR 颜色"""
        return CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def update_thresholds(self, conf: float, iou: float):
        self.conf_threshold = conf
        self.iou_threshold  = iou
