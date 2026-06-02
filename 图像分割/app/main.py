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
import shutil
import mimetypes
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

from app.inference   import DefectDetector, FrameResult, MERGE_IOU_THRESHOLD
from app.visualizer  import draw_detections, image_to_base64
from app.reporter    import generate_excel_report, generate_pdf_report
from app.cause_analyzer import MODEL_NAME as CAUSE_MODEL_NAME


# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent.parent
MODEL_PATH   = BASE_DIR / "pt" / "best.pt"
STATIC_DIR   = BASE_DIR / "static"
REPORTS_DIR  = BASE_DIR / "reports"
UPLOADS_DIR  = BASE_DIR / "uploads"
VIDEO_ASSETS_DIR = UPLOADS_DIR / "video_assets"

REPORTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
VIDEO_ASSETS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────

# session_id -> { results, source_name, annotated_images, created_at }
SESSIONS: dict[str, dict] = {}
ACTIVE_RUNS: dict[str, dict] = {}
ACTIVE_RUNS_LOCK = threading.Lock()

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
    session_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    sid = session_id or str(uuid.uuid4())
    SESSIONS[sid] = {
        "results"          : results,
        "source_name"      : source_name,
        "annotated_images" : annotated_images,
        "created_at"       : time.time(),
    }
    if extra:
        SESSIONS[sid].update(extra)
    # 自动清理超过 1 小时的会话
    _cleanup_old_sessions()
    return sid


def _cleanup_old_sessions(max_age_sec: int = 3600):
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items()
               if now - s["created_at"] > max_age_sec]
    for sid in expired:
        asset_dir = SESSIONS.get(sid, {}).get("asset_dir")
        if asset_dir:
            shutil.rmtree(asset_dir, ignore_errors=True)
        del SESSIONS[sid]


def _get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return SESSIONS[session_id]


def _register_run(run_id: Optional[str], **state) -> str:
    rid = (run_id or str(uuid.uuid4())).strip()
    with ACTIVE_RUNS_LOCK:
        existing = ACTIVE_RUNS.get(rid, {})
        cancel_event = existing.get("cancel_event") or threading.Event()
        existing.update(state)
        existing["cancel_event"] = cancel_event
        ACTIVE_RUNS[rid] = existing
    return rid


def _get_run(run_id: Optional[str]) -> Optional[dict]:
    if not run_id:
        return None
    with ACTIVE_RUNS_LOCK:
        return ACTIVE_RUNS.get(run_id)


def _set_run_state(run_id: Optional[str], **state):
    if not run_id:
        return
    with ACTIVE_RUNS_LOCK:
        if run_id in ACTIVE_RUNS:
            ACTIVE_RUNS[run_id].update(state)


def _cleanup_run_assets(run_state: dict):
    session_id = run_state.get("session_id")
    if session_id and session_id in SESSIONS:
        asset_dir = SESSIONS[session_id].get("asset_dir")
        if asset_dir:
            shutil.rmtree(asset_dir, ignore_errors=True)
        del SESSIONS[session_id]

    for key in ("tmp_path", "asset_dir"):
        value = run_state.get(key)
        if not value:
            continue
        path = Path(value)
        if key == "tmp_path":
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path, ignore_errors=True)


def _finish_run(run_id: Optional[str], cleanup: bool = False):
    if not run_id:
        return
    with ACTIVE_RUNS_LOCK:
        run_state = ACTIVE_RUNS.pop(run_id, None)
    if cleanup and run_state:
        _cleanup_run_assets(run_state)


def _check_cancelled(cancel_event: Optional[threading.Event]):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("run cancelled")


def _collect_video_instance_images(session: dict) -> list[dict]:
    asset_dir = session.get("asset_dir")
    if not asset_dir:
        return []
    base = Path(asset_dir).resolve()
    items = []
    for result in session.get("results", []):
        for det in result.detections:
            if not det.best_crop_url:
                continue
            rel = det.best_crop_url.split("/api/video-assets/", 1)[-1]
            parts = rel.split("/", 1)
            if len(parts) != 2:
                continue
            target = (base / parts[1]).resolve()
            if base not in target.parents and target != base:
                continue
            items.append({
                "path": str(target),
                "det_id": det.det_id,
                "track_id": det.track_id,
                "class_name": det.class_name,
                "frame_index": det.best_frame_index if det.best_frame_index is not None else det.first_frame,
                "first_time": det.first_time,
                "last_time": det.last_time,
            })
    return items


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
        "cause_model" : CAUSE_MODEL_NAME,
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
            "merge_iou_threshold": MERGE_IOU_THRESHOLD,
            "cause_model": CAUSE_MODEL_NAME,
            "cause_analysis_enabled": True,
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

    # 生成原始分辨率标注图
    ann_img  = draw_detections(image, result)
    img_b64  = image_to_base64(ann_img, "jpeg")

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
    run_id: Optional[str]   = Form(None),
):
    """
    批量上传多张图像，返回汇总统计及各图结果。
    """
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="单次批量最多 50 张图")

    rid = _register_run(run_id, kind="batch")
    run_state = _get_run(rid) or {}
    cancel_event = run_state.get("cancel_event")

    payloads = []
    for f in files:
        payloads.append((f.filename or "image", await f.read()))

    def run_batch():
        _check_cancelled(cancel_event)
        detector = get_detector()
        detector.update_thresholds(conf, iou)

        all_results: list[FrameResult] = []
        all_ann: list                   = []
        thumbnails_b64: list[str]       = []

        for idx, (_name, data) in enumerate(payloads):
            _check_cancelled(cancel_event)
            try:
                image = _decode_upload(data)
            except Exception:
                continue

            result  = detector.detect_image(image, frame_index=idx)
            _check_cancelled(cancel_event)
            ann_img = draw_detections(image, result)

            all_results.append(result)
            all_ann.append(ann_img)
            thumbnails_b64.append(image_to_base64(ann_img, "jpeg"))

        _check_cancelled(cancel_event)
        session_id = _make_session(all_results, f"{len(files)} images", all_ann)
        _set_run_state(rid, session_id=session_id)

        total_defs = sum(r.total_defects for r in all_results)
        by_class: dict[str, int] = {}
        for r in all_results:
            for cls, cnt in r.by_class.items():
                by_class[cls] = by_class.get(cls, 0) + cnt

        return {
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
        }

    try:
        data = await asyncio.to_thread(run_batch)
        return JSONResponse(data)
    except InterruptedError:
        _finish_run(rid, cleanup=True)
        raise HTTPException(status_code=499, detail="批处理已中断")
    except Exception as e:
        _finish_run(rid, cleanup=True)
        raise HTTPException(status_code=500, detail=f"批量推理失败: {e}")
    finally:
        _finish_run(rid)


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
    run_id          : Optional[str] = Form(None),
):
    """
    上传视频文件，逐帧检测，返回汇总结果。
    大视频建议使用 WebSocket 接口 /ws/detect/video 获取实时进度。
    """
    if max_frames > 500:
        max_frames = 500

    session_id = str(uuid.uuid4())
    asset_dir = VIDEO_ASSETS_DIR / session_id
    asset_url_prefix = f"/api/video-assets/{session_id}"
    rid = _register_run(run_id, kind="video", session_id=session_id, asset_dir=str(asset_dir))
    run_state = _get_run(rid) or {}
    cancel_event = run_state.get("cancel_event")

    # 保存临时文件
    suffix = Path(file.filename or "video.mp4").suffix
    tmp_path = UPLOADS_DIR / f"{session_id}{suffix}"
    _set_run_state(rid, tmp_path=str(tmp_path))

    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        _check_cancelled(cancel_event)

        def run_video():
            _check_cancelled(cancel_event)
            detector = get_detector()
            detector.update_thresholds(conf, iou)

            return detector.detect_video_tracking(
                str(tmp_path),
                output_dir       = asset_dir,
                asset_url_prefix = asset_url_prefix,
                conf             = conf,
                iou              = iou,
                sample_interval  = sample_interval,
                max_frames       = max_frames,
                tracker          = "botsort.yaml",
                cancel_event     = cancel_event,
            )

        video_result = await asyncio.to_thread(run_video)
        _check_cancelled(cancel_event)
    except Exception as e:
        shutil.rmtree(asset_dir, ignore_errors=True)
        _finish_run(rid, cleanup=True)
        if isinstance(e, InterruptedError):
            raise HTTPException(status_code=499, detail="视频处理已中断")
        raise HTTPException(status_code=500, detail=f"视频处理失败: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    results = video_result["results"]
    timeline = video_result["timeline"]
    summary = video_result["summary"]
    _make_session(
        results,
        file.filename or "video",
        [],
        session_id=session_id,
        extra={
            "asset_dir": str(asset_dir),
            "annotated_video_path": video_result.get("annotated_video_path", ""),
            "annotated_video_url": video_result.get("annotated_video_url", ""),
            "timeline": timeline,
            "video_summary": summary,
            "is_video": True,
        },
    )
    _finish_run(rid)

    return JSONResponse({
        "session_id"          : session_id,
        "total_frames"        : summary.get("processed_frames", 0),
        "annotated_video_url" : video_result.get("annotated_video_url", ""),
        "timeline"            : timeline,
        "results"             : [r.to_dict() for r in results],
        "summary"             : summary,
    })


# ─────────────────────────────────────────────
# WebSocket：实时视频帧检测
# ─────────────────────────────────────────────

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """
    WebSocket 分阶段检测接口。
    客户端发送 base64 编码的图像帧，服务端先返回 YOLO+实例合并结果，
    再返回补充 CLIP/SigLIP 成因分析后的完整结果。

    消息格式（客户端 → 服务端）:
        {"image": "<base64>", "conf": 0.25, "iou": 0.45}

    消息格式（服务端 → 客户端）:
        {"stage": "segmentation_done", "image_b64": "...", "result": {...}}
        {"stage": "analysis_done", "result": {...}}
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
                rid = _register_run(msg.get("run_id"), kind="websocket")
                run_state = _get_run(rid) or {}
                cancel_event = run_state.get("cancel_event")

                # 解码 base64 图像
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                image = _decode_upload(img_bytes)

                detector.update_thresholds(conf, iou)
                result  = detector.detect_image(image, analyze_causes=False)
                _check_cancelled(cancel_event)
                ann_img = draw_detections(image, result)
                img_out = image_to_base64(ann_img, "jpeg")
                session_id = _make_session(
                    results          = [result],
                    source_name      = msg.get("source_name", "websocket-image"),
                    annotated_images = [ann_img],
                )
                _set_run_state(rid, session_id=session_id)

                await websocket.send_json({
                    "stage"    : "segmentation_done",
                    "session_id": session_id,
                    "source_name": msg.get("source_name", "websocket-image"),
                    "image_b64": img_out,
                    "result"   : result.to_dict(),
                    "summary"  : {
                        "total_defects"   : result.total_defects,
                        "by_class"        : result.by_class,
                        "inference_time_ms": round(result.inference_time_ms, 2),
                        "image_size"      : f"{image.shape[1]}x{image.shape[0]}",
                    },
                })

                _check_cancelled(cancel_event)
                detector.add_cause_analysis(image, result)
                await websocket.send_json({
                    "stage"    : "analysis_done",
                    "session_id": session_id,
                    "source_name": msg.get("source_name", "websocket-image"),
                    "result"   : result.to_dict(),
                    "summary"  : {
                        "total_defects"   : result.total_defects,
                        "by_class"        : result.by_class,
                        "inference_time_ms": round(result.inference_time_ms, 2),
                        "image_size"      : f"{image.shape[1]}x{image.shape[0]}",
                    },
                })
                _finish_run(rid)
            except Exception as e:
                if "rid" in locals():
                    _finish_run(rid)
                await websocket.send_json({"stage": "error", "error": str(e)})

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
            instance_image_paths = _collect_video_instance_images(session) if session.get("is_video") else None,
            video_summary    = session.get("video_summary") if session.get("is_video") else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 生成失败: {e}")

    filename = f"defect_report_{session_id[:8]}.pdf"
    return StreamingResponse(
        io.BytesIO(raw),
        media_type = "application/pdf",
        headers    = {"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/video-assets/{session_id}/{asset_path:path}")
async def get_video_asset(session_id: str, asset_path: str):
    """读取视频检测生成的标注视频或最佳实例 crop。"""
    session = _get_session(session_id)
    asset_dir = session.get("asset_dir")
    if not asset_dir:
        raise HTTPException(status_code=404, detail="视频资产不存在")

    base = Path(asset_dir).resolve()
    target = (base / asset_path).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="非法资产路径")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="资产文件不存在")
    media_type = "video/mp4" if target.suffix.lower() == ".mp4" else mimetypes.guess_type(str(target))[0]
    return FileResponse(str(target), media_type=media_type)


# ─────────────────────────────────────────────
# 路由：会话管理
# ─────────────────────────────────────────────

@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    run_state = _get_run(run_id)
    if not run_state:
        return {"status": "not_found"}

    cancel_event = run_state.get("cancel_event")
    if cancel_event:
        cancel_event.set()
    _cleanup_run_assets(run_state)
    return {"status": "cancelling"}


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
        asset_dir = SESSIONS[session_id].get("asset_dir")
        if asset_dir:
            shutil.rmtree(asset_dir, ignore_errors=True)
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
    img_b64  = image_to_base64(ann_img, "jpeg")

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
