"""
基础设施外观缺陷智能检测系统 - FastAPI 后端
端口: 8000
"""

import os
import io
import uuid
import time
import json
import asyncio
import threading
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fastapi import (
    FastAPI, File, UploadFile, Form, HTTPException,
    WebSocket, WebSocketDisconnect, BackgroundTasks,
)
from fastapi.responses import (
    HTMLResponse, FileResponse, StreamingResponse, JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.inference   import DefectDetector, FrameResult
from app.visualizer  import draw_detections, image_to_base64, thumbnail
from app.reporter    import generate_excel_report, generate_pdf_report


# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent.parent
MODEL_PATH   = BASE_DIR / "pt" / "yolov8n-seg-cracks-joints.pt"
STATIC_DIR   = BASE_DIR / "static"
REPORTS_DIR  = BASE_DIR / "reports"
UPLOADS_DIR  = BASE_DIR / "uploads"

REPORTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────

# session_id -> { results, source_name, annotated_images, created_at }
SESSIONS: dict[str, dict] = {}

# 检测器单例（延迟加载）
_detector: Optional[DefectDetector] = None
_detector_lock = threading.Lock()


def get_detector(conf: float = 0.25, iou: float = 0.45) -> DefectDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = DefectDetector(
                model_path     = str(MODEL_PATH),
                conf_threshold = conf,
                iou_threshold  = iou,
            )
        return _detector


# ─────────────────────────────────────────────
# FastAPI 应用
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "基础设施外观缺陷智能检测系统",
    description = "Infrastructure Defect Detection API powered by YOLOv8-Seg",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# 静态文件服务
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _decode_upload(data: bytes) -> np.ndarray:
    """将上传的图像字节解码为 BGR numpy array"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像，请确认文件格式")
    return img


def _make_session(
    results: list[FrameResult],
    source_name: str,
    annotated_images: list,
) -> str:
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "results"          : results,
        "source_name"      : source_name,
        "annotated_images" : annotated_images,
        "created_at"       : time.time(),
    }
    # 自动清理超过 1 小时的会话
    _cleanup_old_sessions()
    return sid


def _cleanup_old_sessions(max_age_sec: int = 3600):
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items()
               if now - s["created_at"] > max_age_sec]
    for sid in expired:
        del SESSIONS[sid]


def _get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return SESSIONS[session_id]


# ─────────────────────────────────────────────
# 路由：前端页面
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h1>前端文件缺失，请检查 static/index.html</h1>", status_code=500)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# 路由：系统状态
# ─────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status"      : "ok",
        "model_exists": MODEL_PATH.exists(),
        "model_path"  : str(MODEL_PATH),
    }


@app.get("/api/model/info")
async def model_info():
    try:
        d = get_detector()
        return {
            "model_path"  : str(MODEL_PATH),
            "class_names" : d.class_names,
            "conf_threshold": d.conf_threshold,
            "iou_threshold" : d.iou_threshold,
            "is_loaded"   : d.is_loaded,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# 路由：图像检测
# ─────────────────────────────────────────────

@app.post("/api/detect/image")
async def detect_image(
    file      : UploadFile = File(...),
    conf      : float      = Form(0.25),
    iou       : float      = Form(0.45),
    thumbnail_size: int    = Form(1024),
):
    """
    上传单张图像，返回检测结果 + 标注图 base64。
    """
    try:
        data  = await file.read()
        image = _decode_upload(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像解码失败: {e}")

    try:
        detector = get_detector()
        detector.update_thresholds(conf, iou)
        result   = detector.detect_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}")

    # 生成标注图（缩略图用于传输）
    ann_img  = draw_detections(image, result)
    disp_img = thumbnail(ann_img, thumbnail_size)
    img_b64  = image_to_base64(disp_img, "jpeg")

    # 存入会话
    session_id = _make_session(
        results          = [result],
        source_name      = file.filename or "image",
        annotated_images = [ann_img],
    )

    return JSONResponse({
        "session_id"   : session_id,
        "source_name"  : file.filename,
        "image_b64"    : img_b64,
        "result"       : result.to_dict(),
        "summary": {
            "total_defects"   : result.total_defects,
            "by_class"        : result.by_class,
            "inference_time_ms": round(result.inference_time_ms, 2),
            "image_size"      : f"{image.shape[1]}x{image.shape[0]}",
        },
    })


# ─────────────────────────────────────────────
# 路由：批量图像检测
# ─────────────────────────────────────────────

@app.post("/api/detect/batch")
async def detect_batch(
    files: list[UploadFile] = File(...),
    conf : float            = Form(0.25),
    iou  : float            = Form(0.45),
):
    """
    批量上传多张图像，返回汇总统计及各图结果。
    """
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="单次批量最多 50 张图")

    detector = get_detector()
    detector.update_thresholds(conf, iou)

    all_results: list[FrameResult] = []
    all_ann: list                   = []
    thumbnails_b64: list[str]       = []

    for idx, f in enumerate(files):
        try:
            data  = await f.read()
            image = _decode_upload(data)
        except Exception:
            continue

        result  = detector.detect_image(image, frame_index=idx)
        ann_img = draw_detections(image, result)
        disp    = thumbnail(ann_img, 512)

        all_results.append(result)
        all_ann.append(ann_img)
        thumbnails_b64.append(image_to_base64(disp, "jpeg"))

    session_id = _make_session(all_results, f"{len(files)} images", all_ann)

    total_defs = sum(r.total_defects for r in all_results)
    by_class: dict[str, int] = {}
    for r in all_results:
        for cls, cnt in r.by_class.items():
            by_class[cls] = by_class.get(cls, 0) + cnt

    return JSONResponse({
        "session_id"    : session_id,
        "processed"     : len(all_results),
        "thumbnails_b64": thumbnails_b64,
        "results"       : [r.to_dict() for r in all_results],
        "summary": {
            "total_defects"  : total_defs,
            "by_class"       : by_class,
            "avg_inference_ms": round(
                sum(r.inference_time_ms for r in all_results) / max(len(all_results), 1), 2
            ),
        },
    })


# ─────────────────────────────────────────────
# 路由：视频检测（带进度）
# ─────────────────────────────────────────────

@app.post("/api/detect/video")
async def detect_video(
    file            : UploadFile = File(...),
    conf            : float      = Form(0.25),
    iou             : float      = Form(0.45),
    sample_interval : int        = Form(5),
    max_frames      : int        = Form(200),
):
    """
    上传视频文件，逐帧检测，返回汇总结果。
    大视频建议使用 WebSocket 接口 /ws/detect/video 获取实时进度。
    """
    if max_frames > 500:
        max_frames = 500

    # 保存临时文件
    suffix = Path(file.filename or "video.mp4").suffix
    tmp_path = UPLOADS_DIR / f"{uuid.uuid4()}{suffix}"

    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        detector = get_detector()
        detector.update_thresholds(conf, iou)

        results = detector.detect_video(
            str(tmp_path),
            conf            = conf,
            iou             = iou,
            sample_interval = sample_interval,
            max_frames      = max_frames,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频处理失败: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # 为有缺陷的帧生成标注图（最多 20 帧）
    ann_results   = [r for r in results if r.total_defects > 0][:20]
    session_id = _make_session(results, file.filename or "video", [])

    total_defs = sum(r.total_defects for r in results)
    by_class: dict[str, int] = {}
    for r in results:
        for cls, cnt in r.by_class.items():
            by_class[cls] = by_class.get(cls, 0) + cnt

    return JSONResponse({
        "session_id"  : session_id,
        "total_frames": len(results),
        "results"     : [r.to_dict() for r in results],
        "summary": {
            "total_defects"   : total_defs,
            "defective_frames": len([r for r in results if r.total_defects > 0]),
            "by_class"        : by_class,
            "avg_inference_ms": round(
                sum(r.inference_time_ms for r in results) / max(len(results), 1), 2
            ),
        },
    })


# ─────────────────────────────────────────────
# WebSocket：实时视频帧检测
# ─────────────────────────────────────────────

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """
    WebSocket 实时检测接口。
    客户端发送 base64 编码的图像帧，服务端返回检测结果 JSON。

    消息格式（客户端 → 服务端）:
        {"image": "<base64>", "conf": 0.25, "iou": 0.45}

    消息格式（服务端 → 客户端）:
        {"image_b64": "<base64>", "result": {...}, "summary": {...}}
    """
    await websocket.accept()
    detector = get_detector()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg   = json.loads(raw)
                b64   = msg.get("image", "")
                conf  = float(msg.get("conf", 0.25))
                iou   = float(msg.get("iou",  0.45))

                # 解码 base64 图像
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                image = _decode_upload(img_bytes)

                detector.update_thresholds(conf, iou)
                result  = detector.detect_image(image)
                ann_img = draw_detections(image, result)
                disp    = thumbnail(ann_img, 640)
                img_out = image_to_base64(disp, "jpeg")

                await websocket.send_json({
                    "image_b64": img_out,
                    "result"   : result.to_dict(),
                    "summary"  : {
                        "total_defects"   : result.total_defects,
                        "by_class"        : result.by_class,
                        "inference_time_ms": round(result.inference_time_ms, 2),
                    },
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────
# 路由：报告导出
# ─────────────────────────────────────────────

@app.get("/api/report/{session_id}/excel")
async def download_excel(session_id: str):
    """下载 Excel 检测报告"""
    session = _get_session(session_id)
    try:
        raw = generate_excel_report(
            results     = session["results"],
            source_name = session["source_name"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel 生成失败: {e}")

    filename = f"defect_report_{session_id[:8]}.xlsx"
    return StreamingResponse(
        io.BytesIO(raw),
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers    = {"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/report/{session_id}/pdf")
async def download_pdf(session_id: str, include_images: bool = True):
    """下载 PDF 检测报告"""
    session = _get_session(session_id)
    ann_imgs = session["annotated_images"] if include_images else []
    try:
        raw = generate_pdf_report(
            results          = session["results"],
            source_name      = session["source_name"],
            annotated_images = ann_imgs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 生成失败: {e}")

    filename = f"defect_report_{session_id[:8]}.pdf"
    return StreamingResponse(
        io.BytesIO(raw),
        media_type = "application/pdf",
        headers    = {"Content-Disposition": f"attachment; filename={filename}"},
    )


# ─────────────────────────────────────────────
# 路由：会话管理
# ─────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    return {
        "sessions": [
            {
                "session_id" : sid,
                "source_name": s["source_name"],
                "created_at" : s["created_at"],
                "total_frames": len(s["results"]),
                "total_defects": sum(r.total_defects for r in s["results"]),
            }
            for sid, s in SESSIONS.items()
        ]
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"status": "ok"}


# ─────────────────────────────────────────────
# 路由：数据集浏览（调试用）
# ─────────────────────────────────────────────

@app.get("/api/dataset/samples")
async def dataset_samples(limit: int = 12):
    """返回数据集测试集图像列表"""
    test_dir = BASE_DIR / "crack-seg" / "test" / "images"
    if not test_dir.exists():
        return {"images": []}
    imgs = sorted(test_dir.glob("*.jpg"))[:limit]
    return {"images": [str(p.name) for p in imgs]}


@app.get("/api/dataset/image/{filename}")
async def dataset_image(filename: str):
    """返回数据集中的原始图像"""
    img_path = BASE_DIR / "crack-seg" / "test" / "images" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="图像不存在")
    return FileResponse(str(img_path))


@app.post("/api/dataset/detect/{filename}")
async def dataset_detect(
    filename : str,
    conf     : float = Form(0.25),
    iou      : float = Form(0.45),
):
    """对数据集中指定图像执行检测"""
    img_path = BASE_DIR / "crack-seg" / "test" / "images" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="图像不存在")

    image    = cv2.imread(str(img_path))
    detector = get_detector()
    detector.update_thresholds(conf, iou)
    result   = detector.detect_image(image)
    ann_img  = draw_detections(image, result)
    disp     = thumbnail(ann_img, 1024)
    img_b64  = image_to_base64(disp, "jpeg")

    session_id = _make_session([result], filename, [ann_img])

    return JSONResponse({
        "session_id": session_id,
        "image_b64" : img_b64,
        "result"    : result.to_dict(),
        "summary": {
            "total_defects"   : result.total_defects,
            "by_class"        : result.by_class,
            "inference_time_ms": round(result.inference_time_ms, 2),
        },
    })
