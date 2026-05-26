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


# ──────────────────────────────────────────────
# 模型封装
# ──────────────────────────────────────────────

class DefectDetector:
    """
    基础设施外观缺陷检测器（YOLOv8-Seg）

    使用方法::

        detector = DefectDetector("./pt/yolov8n-seg-cracks-joints.pt")
        result = detector.detect_image(cv2_img)
    """

    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str = "./pt/yolov8n-seg-cracks-joints.pt",
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

        detections = self._parse_results(results[0], image.shape)

        return FrameResult(
            frame_index    = frame_index,
            timestamp_sec  = timestamp_sec,
            image_shape    = list(image.shape),
            detections     = detections,
            inference_time_ms = inference_ms,
        )

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

    def _parse_results(self, result, image_shape: tuple) -> list[Detection]:
        """将 ultralytics Result 对象解析为 Detection 列表"""
        detections: list[Detection] = []
        H, W = image_shape[:2]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes     = result.boxes
        has_masks = result.masks is not None

        for i in range(len(boxes)):
            # ── 基础信息 ──
            cls_id  = int(boxes.cls[i].item())
            conf    = float(boxes.conf[i].item())
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

            # ── 边界框（像素） ──
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            bw, bh  = x2 - x1, y2 - y1
            cx, cy  = x1 + bw // 2, y1 + bh // 2
            bbox_area = bw * bh

            # ── 边界框（归一化） ──
            bbox_norm = [
                round(x1 / W, 4), round(y1 / H, 4),
                round(x2 / W, 4), round(y2 / H, 4),
            ]

            # ── 掩码信息 ──
            mask_area  = bbox_area
            mask_poly  = []

            if has_masks and i < len(result.masks.data):
                mask_np = result.masks.data[i].cpu().numpy()

                # 掩码可能被缩放到模型输入尺寸，需 resize 回原图
                if mask_np.shape != (H, W):
                    mask_np = cv2.resize(
                        mask_np, (W, H), interpolation=cv2.INTER_LINEAR
                    )
                mask_bin  = (mask_np > 0.5).astype(np.uint8)
                mask_area = int(mask_bin.sum())

                # 提取最大轮廓用于报告
                contours, _ = cv2.findContours(
                    mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    # 简化轮廓点数（最多 20 个点）
                    epsilon = 0.02 * cv2.arcLength(largest, True)
                    approx  = cv2.approxPolyDP(largest, epsilon, True)
                    mask_poly = approx.reshape(-1, 2).tolist()

            detections.append(Detection(
                det_id      = i,
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
                has_mask    = has_masks,
                mask_polygon = mask_poly,
            ))

        # 按置信度降序排列
        detections.sort(key=lambda d: d.confidence, reverse=True)
        # 重新分配 ID
        for idx, d in enumerate(detections):
            d.det_id = idx + 1

        return detections

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
