"""
推理模块 - YOLOv8 分割模型封装
支持图像推理、视频逐帧推理、批量推理
"""

import time
import cv2
import numpy as np
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
    features: dict = field(default_factory=dict)
    source_indices: list = field(default_factory=list)
    cause_analysis: dict = field(default_factory=dict)

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
        for det in frame_result.detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            crop = image[y1:y2, x1:x2]
            det.cause_analysis = analyzer.analyze_crop(crop)
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
