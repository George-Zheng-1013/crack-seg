"""
可视化模块 - 在原图上绘制检测结果
支持：分割掩码叠加、边界框、类别标签、置信度、序号
"""

import cv2
import numpy as np
from typing import Optional

from app.inference import Detection, FrameResult, CLASS_COLORS_BGR

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

MASK_ALPHA = 0.40  # 掩码叠加透明度
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2
FONT_THICKNESS = 4
BOX_THICKNESS = 2
LABEL_PADDING = 4  # 标签内边距 px


# ──────────────────────────────────────────────
# 主可视化函数
# ──────────────────────────────────────────────


def draw_detections(
    image: np.ndarray,
    result: FrameResult,
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
    draw_index: bool = True,
    mask_alpha: float = MASK_ALPHA,
) -> np.ndarray:
    """
    在图像上绘制检测结果，返回新的标注图像（不修改原图）。

    :param image:      BGR numpy array
    :param result:     FrameResult（来自 DefectDetector.detect_image）
    :param draw_masks: 是否绘制分割掩码叠加
    :param draw_boxes: 是否绘制边界框
    :param draw_labels:是否绘制类别+置信度标签
    :param draw_index: 是否在框内绘制序号
    :param mask_alpha: 掩码透明度 (0~1)
    :return: 标注后的 BGR numpy array
    """
    vis = image.copy()
    detections = result.detections

    if not detections:
        return _draw_no_detection_hint(vis)

    # ── 1. 绘制掩码（先画，在框之下）──────────
    if draw_masks:
        vis = _draw_masks(vis, detections, mask_alpha)

    # ── 2. 绘制边界框 ──────────────────────────
    if draw_boxes:
        for det in detections:
            color = _get_color(det.class_id)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # ── 3. 绘制标签 ────────────────────────────
    if draw_labels:
        for det in detections:
            color = _get_color(det.class_id)
            x1, y1 = det.bbox[0], det.bbox[1]
            idx_str = f"#{det.det_id} " if draw_index else ""
            label = f"{idx_str}{det.class_name} {det.confidence:.2f}"
            _draw_label(vis, label, x1, y1, color)

    # ── 4. 绘制信息角标 ────────────────────────
    _draw_summary_overlay(vis, result)

    return vis


def draw_detections_minimal(
    image: np.ndarray,
    result: FrameResult,
) -> np.ndarray:
    """轻量版：只绘制掩码+框，无标签（用于视频帧缩略图）"""
    return draw_detections(
        image,
        result,
        draw_masks=True,
        draw_boxes=True,
        draw_labels=False,
        draw_index=False,
    )


# ──────────────────────────────────────────────
# 内部工具函数
# ──────────────────────────────────────────────


def _get_color(class_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]


def _draw_masks(
    vis: np.ndarray,
    detections: list,
    alpha: float,
) -> np.ndarray:
    """将每个检测目标的掩码以半透明彩色叠加在图像上"""
    H, W = vis.shape[:2]
    overlay = vis.copy()

    for det in detections:
        if not det.has_mask or not det.mask_polygon:
            # 无掩码时用边界框矩形替代
            x1, y1, x2, y2 = det.bbox
            color = _get_color(det.class_id)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            continue

        pts = np.array(det.mask_polygon, dtype=np.int32)
        if pts.ndim == 2:
            pts = pts.reshape((-1, 1, 2))
        color = _get_color(det.class_id)
        cv2.fillPoly(overlay, [pts], color)

    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def _draw_label(
    vis: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
):
    """在 (x, y) 处绘制带背景的标签"""
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    pad = LABEL_PADDING

    # 标签背景框
    bg_x1 = x
    bg_y1 = max(0, y - th - pad * 2 - baseline)
    bg_x2 = x + tw + pad * 2
    bg_y2 = max(th + pad * 2, y)

    # 半透明背景
    sub = vis[bg_y1:bg_y2, bg_x1:bg_x2]
    if sub.size > 0:
        bg = np.full_like(sub, color)
        blended = cv2.addWeighted(bg, 0.75, sub, 0.25, 0)
        vis[bg_y1:bg_y2, bg_x1:bg_x2] = blended

    # 文字（白色）
    tx = x + pad
    ty = max(th, y - pad - baseline)
    cv2.putText(
        vis,
        text,
        (tx, ty),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _draw_summary_overlay(vis: np.ndarray, result: FrameResult):
    """在左上角绘制统计信息：总数 + 推理时间"""
    H, W = vis.shape[:2]
    lines = [
        f"Total: {result.total_defects}",
        f"Time:  {result.inference_time_ms:.1f} ms",
    ]
    for cls_name, cnt in result.by_class.items():
        lines.append(f"  {cls_name}: {cnt}")

    y_offset = 10
    for line in lines:
        # 使用常量计算文字尺寸
        (tw, th), baseline = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)
        pad = LABEL_PADDING

        x1, y1 = 5, y_offset - 2
        x2, y2 = 5 + tw + pad * 2, y_offset + th + pad * 2

        # 更强的不透明背景
        sub = vis[y1:y2, x1:x2]
        if sub.size > 0:
            bg = np.full_like(sub, (0, 0, 0))
            vis[y1:y2, x1:x2] = cv2.addWeighted(bg, 0.85, sub, 0.15, 0)

        # 白色文字，使用与测量一致的参数
        tx = x1 + pad
        ty = y_offset + th
        cv2.putText(
            vis,
            line,
            (tx, ty),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        y_offset += th + 8


def _draw_no_detection_hint(vis: np.ndarray) -> np.ndarray:
    """当没有检测结果时，在图像中央显示提示"""
    H, W = vis.shape[:2]
    text = "No defects detected"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    cv2.putText(
        vis,
        text,
        ((W - tw) // 2, (H + th) // 2),
        FONT,
        0.8,
        (0, 200, 100),
        2,
        cv2.LINE_AA,
    )
    return vis


# ──────────────────────────────────────────────
# 视频帧 GIF / 缩略图工具
# ──────────────────────────────────────────────


def encode_image_to_jpeg_bytes(image: np.ndarray, quality: int = 100) -> bytes:
    """将 BGR numpy array 编码为 JPEG bytes"""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("图像编码失败")
    return buf.tobytes()


def encode_image_to_png_bytes(image: np.ndarray) -> bytes:
    """将 BGR numpy array 编码为 PNG bytes"""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("图像编码失败")
    return buf.tobytes()


def image_to_base64(image: np.ndarray, fmt: str = "jpeg") -> str:
    """将图像编码为 base64 字符串（用于 API 响应 / HTML 展示）"""
    import base64

    if fmt == "jpeg":
        data = encode_image_to_jpeg_bytes(image)
        mime = "image/jpeg"
    else:
        data = encode_image_to_png_bytes(image)
        mime = "image/png"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def thumbnail(image: np.ndarray, max_side: int = 800) -> np.ndarray:
    """等比缩放，确保长边不超过 max_side px"""
    H, W = image.shape[:2]
    if max(H, W) <= max_side:
        return image
    scale = max_side / max(H, W)
    new_W = int(W * scale)
    new_H = int(H * scale)
    return cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)
