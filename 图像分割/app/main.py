"""
鍩虹璁炬柦澶栬缂洪櫡鏅鸿兘妫€娴嬬郴缁?- FastAPI 鍚庣
绔彛: 8000
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
    HTMLResponse, FileResponse, StreamingResponse, JSONResponse, Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.inference   import (
    DefectDetector, FrameResult, MERGE_IOU_THRESHOLD,
    make_detection_crops, sanitize_name,
)
from app.visualizer  import draw_detections, image_to_base64
from app.reporter    import generate_excel_report, generate_pdf_report
from app.cause_analyzer import MODEL_NAME as CAUSE_MODEL_NAME, get_cause_analyzer


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺緞閰嶇疆
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

BASE_DIR     = Path(__file__).parent.parent
MODEL_PATH   = BASE_DIR / "pt" / "best.pt"
STATIC_DIR   = BASE_DIR / "static"
REPORTS_DIR  = BASE_DIR / "reports"
UPLOADS_DIR  = BASE_DIR / "uploads"
VIDEO_ASSETS_DIR = UPLOADS_DIR / "video_assets"
SESSION_ASSETS_DIR = UPLOADS_DIR / "session_assets"

REPORTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
VIDEO_ASSETS_DIR.mkdir(exist_ok=True)
SESSION_ASSETS_DIR.mkdir(exist_ok=True)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鍏ㄥ眬鐘舵€?# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

# session_id -> { results, source_name, annotated_images, created_at }
SESSIONS: dict[str, dict] = {}
ACTIVE_RUNS: dict[str, dict] = {}
ACTIVE_RUNS_LOCK = threading.Lock()

# 妫€娴嬪櫒鍗曚緥锛堝欢杩熷姞杞斤級
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


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# FastAPI 搴旂敤
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

app = FastAPI(
    title       = "Infrastructure Defect Detection System",
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

# Static file service.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 宸ュ叿鍑芥暟
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _decode_upload(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("鏃犳硶瑙ｇ爜鍥惧儚锛岃纭鏂囦欢鏍煎紡")
    return img


def _decode_data_url_bytes(value: str) -> bytes:
    if "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


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
    # Clean up sessions older than 1 hour.
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
        raise HTTPException(status_code=404, detail="session not found or expired")
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


def _write_instance_crops(
    session_id: str,
    results: list[FrameResult],
    source_images: list[np.ndarray],
    asset_dir: Path,
    asset_url_prefix: str,
) -> None:
    crops_dir = asset_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for result, image in zip(results, source_images):
        frame_index = int(getattr(result, "frame_index", 0) or 0)
        for det in result.detections:
            _raw_crop, crop_vis = make_detection_crops(image, det)
            if crop_vis.size == 0:
                continue
            crop_name = (
                f"frame_{frame_index:04d}_det_{int(det.det_id):03d}_"
                f"{sanitize_name(det.class_name)}.jpg"
            )
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            det.best_crop_url = f"{asset_url_prefix}/crops/{crop_name}"


def _batch_payload(
    session_id: str,
    results: list[FrameResult],
    thumbnails_b64: list[str],
) -> dict:
    total_defs = sum(r.total_defects for r in results)
    by_class: dict[str, int] = {}
    for r in results:
        for cls, cnt in r.by_class.items():
            by_class[cls] = by_class.get(cls, 0) + cnt

    return {
        "session_id"    : session_id,
        "processed"     : len(results),
        "thumbnails_b64": thumbnails_b64,
        "results"       : [r.to_dict() for r in results],
        "summary": {
            "total_defects"  : total_defs,
            "by_class"       : by_class,
            "avg_inference_ms": round(
                sum(r.inference_time_ms for r in results) / max(len(results), 1), 2
            ),
        },
    }


def _video_payload(session_id: str, video_result: dict) -> dict:
    summary = video_result["summary"]
    return {
        "session_id"          : session_id,
        "total_frames"        : summary.get("processed_frames", 0),
        "annotated_video_url" : video_result.get("annotated_video_url", ""),
        "timeline"            : video_result["timeline"],
        "results"             : [r.to_dict() for r in video_result["results"]],
        "summary"             : summary,
    }


def _collect_instance_images(session_id: str, session: dict) -> list[dict]:
    asset_dir = session.get("asset_dir")
    if not asset_dir:
        return []
    base = Path(asset_dir).resolve()
    items = []
    for result in session.get("results", []):
        for det in result.detections:
            if not det.best_crop_url:
                continue
            marker = f"/{session_id}/"
            if marker not in det.best_crop_url:
                continue
            rel = det.best_crop_url.split(marker, 1)[1]
            target = (base / rel).resolve()
            if base not in target.parents and target != base:
                continue
            items.append({
                "path": str(target),
                "det_id": det.det_id,
                "track_id": det.track_id,
                "class_name": det.class_name,
                "frame_index": (
                    det.best_frame_index
                    if det.best_frame_index is not None
                    else det.first_frame
                    if det.first_frame is not None
                    else getattr(result, "frame_index", 0)
                ),
                "first_time": det.first_time,
                "last_time": det.last_time,
            })
    return items


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺敱锛氬墠绔〉闈?# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h1>鍓嶇鏂囦欢缂哄け锛岃妫€鏌?static/index.html</h1>", status_code=500)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺敱锛氱郴缁熺姸鎬?# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺敱锛氬浘鍍忔娴?# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@app.post("/api/detect/image")
async def detect_image(
    file      : UploadFile = File(...),
    conf      : float      = Form(0.25),
    iou       : float      = Form(0.45),
):
    """
    涓婁紶鍗曞紶鍥惧儚锛岃繑鍥炴娴嬬粨鏋?+ 鏍囨敞鍥?base64銆?    """
    try:
        data  = await file.read()
        image = _decode_upload(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"鍥惧儚瑙ｇ爜澶辫触: {e}")

    try:
        detector = get_detector()
        detector.update_thresholds(conf, iou)
        result   = detector.detect_image(image, analyze_causes=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"鎺ㄧ悊澶辫触: {e}")

    # 鐢熸垚鍘熷鍒嗚鲸鐜囨爣娉ㄥ浘
    ann_img  = draw_detections(image, result)
    img_b64  = image_to_base64(ann_img, "jpeg")

    # 瀛樺叆浼氳瘽
    session_id = str(uuid.uuid4())
    asset_dir = SESSION_ASSETS_DIR / session_id
    asset_url_prefix = f"/api/session-assets/{session_id}"
    _write_instance_crops(session_id, [result], [image], asset_dir, asset_url_prefix)
    session_id = _make_session(
        results          = [result],
        source_name      = file.filename or "image",
        annotated_images = [ann_img],
        session_id       = session_id,
        extra            = {"asset_dir": str(asset_dir)},
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


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺敱锛氭壒閲忓浘鍍忔娴?# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@app.post("/api/detect/batch")
async def detect_batch(
    files: list[UploadFile] = File(...),
    conf : float            = Form(0.25),
    iou  : float            = Form(0.45),
    run_id: Optional[str]   = Form(None),
):
    """
    鎵归噺涓婁紶澶氬紶鍥惧儚锛岃繑鍥炴眹鎬荤粺璁″強鍚勫浘缁撴灉銆?    """
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="鍗曟鎵归噺鏈€澶?50 寮犲浘")

    rid = _register_run(run_id, kind="batch")
    run_state = _get_run(rid) or {}
    cancel_event = run_state.get("cancel_event")
    session_id = str(uuid.uuid4())
    asset_dir = SESSION_ASSETS_DIR / session_id
    asset_url_prefix = f"/api/session-assets/{session_id}"
    _set_run_state(rid, session_id=session_id, asset_dir=str(asset_dir))

    payloads = []
    for f in files:
        payloads.append((f.filename or "image", await f.read()))

    def run_batch():
        _check_cancelled(cancel_event)
        detector = get_detector()
        detector.update_thresholds(conf, iou)

        all_results: list[FrameResult] = []
        all_ann: list                   = []
        source_images: list[np.ndarray] = []
        thumbnails_b64: list[str]       = []

        for idx, (_name, data) in enumerate(payloads):
            _check_cancelled(cancel_event)
            try:
                image = _decode_upload(data)
            except Exception:
                continue

            result  = detector.detect_image(image, frame_index=idx, analyze_causes=False)
            _check_cancelled(cancel_event)
            ann_img = draw_detections(image, result)

            all_results.append(result)
            all_ann.append(ann_img)
            source_images.append(image)
            thumbnails_b64.append(image_to_base64(ann_img, "jpeg"))

        _check_cancelled(cancel_event)
        detector.add_cause_analysis_batch(source_images, all_results)
        _check_cancelled(cancel_event)
        _write_instance_crops(session_id, all_results, source_images, asset_dir, asset_url_prefix)
        _make_session(
            all_results,
            f"{len(files)} images",
            all_ann,
            session_id=session_id,
            extra={"asset_dir": str(asset_dir)},
        )
        _set_run_state(rid, session_id=session_id)

        return _batch_payload(session_id, all_results, thumbnails_b64)

    try:
        data = await asyncio.to_thread(run_batch)
        return JSONResponse(data)
    except InterruptedError:
        _finish_run(rid, cleanup=True)
        raise HTTPException(status_code=499, detail="鎵瑰鐞嗗凡涓柇")
    except Exception as e:
        _finish_run(rid, cleanup=True)
        raise HTTPException(status_code=500, detail=f"鎵归噺鎺ㄧ悊澶辫触: {e}")
    finally:
        _finish_run(rid)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺敱锛氳棰戞娴嬶紙甯﹁繘搴︼級
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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
    涓婁紶瑙嗛鏂囦欢锛岄€愬抚妫€娴嬶紝杩斿洖姹囨€荤粨鏋溿€?    澶ц棰戝缓璁娇鐢?WebSocket 鎺ュ彛 /ws/detect/video 鑾峰彇瀹炴椂杩涘害銆?    """
    if max_frames > 500:
        max_frames = 500

    session_id = str(uuid.uuid4())
    asset_dir = VIDEO_ASSETS_DIR / session_id
    asset_url_prefix = f"/api/video-assets/{session_id}"
    rid = _register_run(run_id, kind="video", session_id=session_id, asset_dir=str(asset_dir))
    run_state = _get_run(rid) or {}
    cancel_event = run_state.get("cancel_event")

    # 淇濆瓨涓存椂鏂囦欢
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
            raise HTTPException(status_code=499, detail="video processing cancelled")
        raise HTTPException(status_code=500, detail=f"瑙嗛澶勭悊澶辫触: {e}")
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

    return JSONResponse(_video_payload(session_id, video_result))


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# WebSocket锛氬疄鏃惰棰戝抚妫€娴?# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """
    WebSocket 鍒嗛樁娈垫娴嬫帴鍙ｃ€?    瀹㈡埛绔彂閫?base64 缂栫爜鐨勫浘鍍忓抚锛屾湇鍔＄鍏堣繑鍥?YOLO+瀹炰緥鍚堝苟缁撴灉锛?    鍐嶈繑鍥炶ˉ鍏?CLIP/SigLIP 鎴愬洜鍒嗘瀽鍚庣殑瀹屾暣缁撴灉銆?
    娑堟伅鏍煎紡锛堝鎴风 鈫?鏈嶅姟绔級:
        {"image": "<base64>", "conf": 0.25, "iou": 0.45}

    娑堟伅鏍煎紡锛堟湇鍔＄ 鈫?瀹㈡埛绔級:
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

                # 瑙ｇ爜 base64 鍥惧儚
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                image = _decode_upload(img_bytes)

                detector.update_thresholds(conf, iou)
                result  = detector.detect_image(image, analyze_causes=False)
                _check_cancelled(cancel_event)
                ann_img = draw_detections(image, result)
                img_out = image_to_base64(ann_img, "jpeg")
                session_id = str(uuid.uuid4())
                asset_dir = SESSION_ASSETS_DIR / session_id
                asset_url_prefix = f"/api/session-assets/{session_id}"
                _write_instance_crops(session_id, [result], [image], asset_dir, asset_url_prefix)
                session_id = _make_session(
                    results          = [result],
                    source_name      = msg.get("source_name", "websocket-image"),
                    annotated_images = [ann_img],
                    session_id       = session_id,
                    extra            = {"asset_dir": str(asset_dir)},
                )
                _set_run_state(rid, session_id=session_id, asset_dir=str(asset_dir))

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


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺敱锛氭姤鍛婂鍑?# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@app.websocket("/ws/detect/batch")
async def ws_detect_batch(websocket: WebSocket):
    await websocket.accept()
    detector = get_detector()

    try:
        while True:
            raw = await websocket.receive_text()
            rid = None
            sent_segmentation = False
            try:
                msg = json.loads(raw)
                files = msg.get("files", [])
                if len(files) > 50:
                    raise ValueError("single batch supports at most 50 images")

                conf = float(msg.get("conf", 0.25))
                iou = float(msg.get("iou", 0.45))
                rid = _register_run(msg.get("run_id"), kind="batch-websocket")
                run_state = _get_run(rid) or {}
                cancel_event = run_state.get("cancel_event")
                session_id = str(uuid.uuid4())
                asset_dir = SESSION_ASSETS_DIR / session_id
                asset_url_prefix = f"/api/session-assets/{session_id}"
                _set_run_state(rid, session_id=session_id, asset_dir=str(asset_dir))

                detector.update_thresholds(conf, iou)
                all_results: list[FrameResult] = []
                all_ann: list = []
                source_images: list[np.ndarray] = []
                thumbnails_b64: list[str] = []

                for idx, item in enumerate(files):
                    _check_cancelled(cancel_event)
                    image = _decode_upload(_decode_data_url_bytes(str(item.get("image", ""))))
                    result = detector.detect_image(image, frame_index=idx, analyze_causes=False)
                    ann_img = draw_detections(image, result)
                    all_results.append(result)
                    all_ann.append(ann_img)
                    source_images.append(image)
                    thumbnails_b64.append(image_to_base64(ann_img, "jpeg"))

                _check_cancelled(cancel_event)
                _write_instance_crops(session_id, all_results, source_images, asset_dir, asset_url_prefix)
                _make_session(
                    all_results,
                    f"{len(files)} images",
                    all_ann,
                    session_id=session_id,
                    extra={"asset_dir": str(asset_dir)},
                )
                _set_run_state(rid, session_id=session_id)
                await websocket.send_json({"stage": "segmentation_done", **_batch_payload(session_id, all_results, thumbnails_b64)})
                sent_segmentation = True

                _check_cancelled(cancel_event)
                detector.add_cause_analysis_batch(source_images, all_results)
                await websocket.send_json({"stage": "analysis_done", **_batch_payload(session_id, all_results, thumbnails_b64)})
                _finish_run(rid)
            except Exception as e:
                if rid:
                    _finish_run(rid, cleanup=not sent_segmentation)
                await websocket.send_json({"stage": "error", "error": str(e)})

    except WebSocketDisconnect:
        pass


@app.websocket("/ws/detect/video")
async def ws_detect_video(websocket: WebSocket):
    await websocket.accept()
    detector = get_detector()

    try:
        while True:
            raw = await websocket.receive_text()
            rid = None
            tmp_path: Optional[Path] = None
            sent_segmentation = False
            try:
                msg = json.loads(raw)
                conf = float(msg.get("conf", 0.25))
                iou = float(msg.get("iou", 0.45))
                sample_interval = int(msg.get("sample_interval", 5))
                max_frames = min(int(msg.get("max_frames", 200)), 500)
                filename = str(msg.get("filename") or "video.mp4")

                session_id = str(uuid.uuid4())
                asset_dir = VIDEO_ASSETS_DIR / session_id
                asset_url_prefix = f"/api/video-assets/{session_id}"
                rid = _register_run(
                    msg.get("run_id"),
                    kind="video-websocket",
                    session_id=session_id,
                    asset_dir=str(asset_dir),
                )
                run_state = _get_run(rid) or {}
                cancel_event = run_state.get("cancel_event")

                suffix = Path(filename).suffix or ".mp4"
                tmp_path = UPLOADS_DIR / f"{session_id}{suffix}"
                tmp_path.write_bytes(_decode_data_url_bytes(str(msg.get("video", ""))))
                _set_run_state(rid, tmp_path=str(tmp_path))

                detector.update_thresholds(conf, iou)
                video_result = await asyncio.to_thread(
                    detector.detect_video_tracking,
                    str(tmp_path),
                    asset_dir,
                    asset_url_prefix,
                    conf,
                    iou,
                    sample_interval,
                    max_frames,
                    "botsort.yaml",
                    None,
                    cancel_event,
                    False,
                )
                _check_cancelled(cancel_event)

                _make_session(
                    video_result["results"],
                    filename,
                    [],
                    session_id=session_id,
                    extra={
                        "asset_dir": str(asset_dir),
                        "annotated_video_path": video_result.get("annotated_video_path", ""),
                        "annotated_video_url": video_result.get("annotated_video_url", ""),
                        "timeline": video_result["timeline"],
                        "video_summary": video_result["summary"],
                        "is_video": True,
                    },
                )
                await websocket.send_json({"stage": "segmentation_done", **_video_payload(session_id, video_result)})
                sent_segmentation = True

                _check_cancelled(cancel_event)
                analysis_items = video_result.get("_analysis_items", [])
                analysis_detections = video_result.get("_analysis_detections", [])
                if analysis_items:
                    analyses = get_cause_analyzer().analyze_batch(analysis_items)
                    for det, analysis in zip(analysis_detections, analyses):
                        det.cause_analysis = analysis

                await websocket.send_json({"stage": "analysis_done", **_video_payload(session_id, video_result)})
                _finish_run(rid)
            except Exception as e:
                if rid:
                    _finish_run(rid, cleanup=not sent_segmentation)
                await websocket.send_json({"stage": "error", "error": str(e)})
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

    except WebSocketDisconnect:
        pass


@app.get("/api/report/{session_id}/excel")
async def download_excel(session_id: str):
    """Download Excel report."""
    session = _get_session(session_id)
    try:
        raw = generate_excel_report(
            results     = session["results"],
            source_name = session["source_name"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel 鐢熸垚澶辫触: {e}")

    filename = f"defect_report_{session_id[:8]}.xlsx"
    return StreamingResponse(
        io.BytesIO(raw),
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers    = {"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/report/{session_id}/pdf")
async def download_pdf(session_id: str, include_images: bool = True):
    """Download PDF report."""
    session = _get_session(session_id)
    ann_imgs = session["annotated_images"] if include_images else []
    try:
        raw = generate_pdf_report(
            results          = session["results"],
            source_name      = session["source_name"],
            annotated_images = ann_imgs,
            instance_image_paths = _collect_instance_images(session_id, session),
            video_summary    = session.get("video_summary") if session.get("is_video") else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 鐢熸垚澶辫触: {e}")

    filename = f"defect_report_{session_id[:8]}.pdf"
    return StreamingResponse(
        io.BytesIO(raw),
        media_type = "application/pdf",
        headers    = {"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/video-assets/{session_id}/{asset_path:path}")
async def get_video_asset(session_id: str, asset_path: str):
    """Read video assets generated by detection."""
    session = _get_session(session_id)
    asset_dir = session.get("asset_dir")
    if not asset_dir:
        raise HTTPException(status_code=404, detail="video assets not found")

    base = Path(asset_dir).resolve()
    target = (base / asset_path).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="闈炴硶璧勪骇璺緞")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="asset file not found")
    media_type = "video/mp4" if target.suffix.lower() == ".mp4" else mimetypes.guess_type(str(target))[0]
    return FileResponse(str(target), media_type=media_type)


@app.get("/api/session-assets/{session_id}/{asset_path:path}")
async def get_session_asset(session_id: str, asset_path: str):
    """Read image or batch instance crop assets."""
    session = _get_session(session_id)
    asset_dir = session.get("asset_dir")
    if not asset_dir:
        raise HTTPException(status_code=404, detail="session assets not found")

    base = Path(asset_dir).resolve()
    target = (base / asset_path).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="闈炴硶璧勪骇璺緞")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="asset file not found")
    return FileResponse(str(target), media_type=mimetypes.guess_type(str(target))[0])


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺敱锛氫細璇濈鐞?# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺敱锛氭暟鎹泦娴忚锛堣皟璇曠敤锛?# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@app.get("/api/dataset/samples")
async def dataset_samples(limit: int = 12):
    """Return test dataset image names."""
    test_dir = BASE_DIR / "crack-seg" / "test" / "images"
    if not test_dir.exists():
        return {"images": []}
    imgs = sorted(test_dir.glob("*.jpg"))[:limit]
    return {"images": [str(p.name) for p in imgs]}


@app.get("/api/dataset/image/{filename}")
async def dataset_image(filename: str):
    """Return a source image from the dataset."""
    img_path = BASE_DIR / "crack-seg" / "test" / "images" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(img_path))


@app.post("/api/dataset/detect/{filename}")
async def dataset_detect(
    filename : str,
    conf     : float = Form(0.25),
    iou      : float = Form(0.45),
):
    """Run detection for one dataset image."""
    img_path = BASE_DIR / "crack-seg" / "test" / "images" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="image not found")

    image    = cv2.imread(str(img_path))
    detector = get_detector()
    detector.update_thresholds(conf, iou)
    result   = detector.detect_image(image)
    ann_img  = draw_detections(image, result)
    img_b64  = image_to_base64(ann_img, "jpeg")

    session_id = str(uuid.uuid4())
    asset_dir = SESSION_ASSETS_DIR / session_id
    asset_url_prefix = f"/api/session-assets/{session_id}"
    _write_instance_crops(session_id, [result], [image], asset_dir, asset_url_prefix)
    session_id = _make_session(
        [result],
        filename,
        [ann_img],
        session_id=session_id,
        extra={"asset_dir": str(asset_dir)},
    )

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
