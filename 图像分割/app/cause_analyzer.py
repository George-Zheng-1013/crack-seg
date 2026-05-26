"""
CLIP/SigLIP cause analysis wrapper.

The FastAPI/YOLO process imports OpenCV. On this Windows environment, loading
the SigLIP torch model in the same process after cv2 can trigger a native access
violation. To keep the server stable, SigLIP runs in a short-lived worker
subprocess that does not import cv2.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_NAME = "google/siglip-so400m-patch14-384"
PROMPTS_PATH = Path(__file__).with_name("cause_prompts.json")
WORKER_TIMEOUT_SEC = int(os.getenv("CAUSE_WORKER_TIMEOUT_SEC", "240"))
CPU_FALLBACK_TIMEOUT_SEC = int(os.getenv("CAUSE_CPU_FALLBACK_TIMEOUT_SEC", "600"))
WINDOWS_ACCESS_VIOLATION_EXIT = 3221225477


@dataclass
class CauseAnalyzer:
    model_name: str = MODEL_NAME

    def __post_init__(self):
        self._prompt_library = None
        self._prompt_error: str | None = None
        self._prompt_lock = threading.Lock()

    def _load_prompts(self):
        if self._prompt_library is not None or self._prompt_error is not None:
            return

        with self._prompt_lock:
            if self._prompt_library is not None or self._prompt_error is not None:
                return
            try:
                raw = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
                self._prompt_library = {
                    str(class_name).lower(): records
                    for class_name, records in raw.items()
                    if isinstance(records, list)
                }
            except Exception as exc:
                self._prompt_error = str(exc)

    def analyze_crop(
        self, crop_bgr: np.ndarray, class_name: str, top_k: int = 3
    ) -> dict:
        return self.analyze_batch(
            [{"crop_bgr": crop_bgr, "class_name": class_name}],
            top_k=top_k,
        )[0]

    def analyze_batch(self, items: list[dict], top_k: int = 3) -> list[dict]:
        self._load_prompts()
        if self._prompt_library is None:
            return [
                _empty_analysis(self._prompt_error or "prompt library unavailable")
                for _ in items
            ]

        payload_items = []
        fallback_results = []
        payload_index_to_item_index = {}

        for item_index, item in enumerate(items):
            class_key = str(item.get("class_name") or "").lower()
            crop_bgr = item.get("crop_bgr")
            prompt_records = self._prompt_library.get(class_key, [])

            if not prompt_records:
                fallback_results.append(
                    _empty_analysis(f"no prompts for class: {class_key}", class_key)
                )
                continue
            if crop_bgr is None or crop_bgr.size == 0:
                fallback_results.append(_empty_analysis("empty crop", class_key))
                continue

            payload_index_to_item_index[len(payload_items)] = item_index
            fallback_results.append(None)
            payload_items.append(
                {
                    "class_name": class_key,
                    "image_b64": _encode_crop_png(crop_bgr),
                }
            )

        if not payload_items:
            return fallback_results

        worker_results = self._run_worker(payload_items, top_k)
        if len(worker_results) != len(payload_items):
            err = f"worker returned {len(worker_results)} results for {len(payload_items)} inputs"
            worker_results = [_empty_analysis(err) for _ in payload_items]

        for payload_index, result in enumerate(worker_results):
            item_index = payload_index_to_item_index[payload_index]
            fallback_results[item_index] = result

        return fallback_results

    def _run_worker(self, payload_items: list[dict], top_k: int) -> list[dict]:
        app_root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(app_root)

        payload = {
            "model_name": self.model_name,
            "prompts_path": str(PROMPTS_PATH),
            "top_k": top_k,
            "items": payload_items,
        }

        proc = self._run_worker_process(payload, app_root, env, WORKER_TIMEOUT_SEC)
        if isinstance(proc, subprocess.TimeoutExpired):
            return [
                _empty_analysis(f"cause worker timed out after {WORKER_TIMEOUT_SEC}s")
                for _ in payload_items
            ]
        if isinstance(proc, Exception):
            return [
                _empty_analysis(f"cause worker failed: {proc}") for _ in payload_items
            ]

        if proc.returncode != 0:
            if (
                proc.returncode == WINDOWS_ACCESS_VIOLATION_EXIT
                and env.get("CAUSE_MODEL_DEVICE", "auto").lower() != "cpu"
            ):
                cpu_env = env.copy()
                cpu_env["CAUSE_MODEL_DEVICE"] = "cpu"
                cpu_proc = self._run_worker_process(
                    payload,
                    app_root,
                    cpu_env,
                    CPU_FALLBACK_TIMEOUT_SEC,
                )
                if isinstance(cpu_proc, subprocess.TimeoutExpired):
                    return [
                        _empty_analysis(
                            "GPU SigLIP worker crashed with access violation; "
                            f"CPU fallback timed out after {CPU_FALLBACK_TIMEOUT_SEC}s"
                        )
                        for _ in payload_items
                    ]
                if isinstance(cpu_proc, Exception):
                    return [
                        _empty_analysis(
                            "GPU SigLIP worker crashed with access violation; "
                            f"CPU fallback failed: {cpu_proc}"
                        )
                        for _ in payload_items
                    ]
                if cpu_proc.returncode == 0:
                    proc = cpu_proc
                else:
                    cpu_err = (
                        cpu_proc.stderr
                        or cpu_proc.stdout
                        or f"CPU fallback worker exit code {cpu_proc.returncode}"
                    ).strip()
                    return [
                        _empty_analysis(
                            "GPU SigLIP worker crashed with access violation; "
                            f"CPU fallback failed: {cpu_err}"
                        )
                        for _ in payload_items
                    ]
            else:
                err = _worker_error(proc)
                return [_empty_analysis(err) for _ in payload_items]

        if proc.returncode != 0:
            err = _worker_error(proc)
            return [_empty_analysis(err) for _ in payload_items]

        try:
            data = json.loads(proc.stdout)
            return data.get("results", [])
        except Exception as exc:
            err = f"invalid worker output: {exc}; stdout={proc.stdout[:500]}"
            return [_empty_analysis(err) for _ in payload_items]

    def _run_worker_process(
        self,
        payload: dict,
        app_root: Path,
        env: dict,
        timeout_sec: int,
    ):
        try:
            return subprocess.run(
                [sys.executable, "-m", "app.cause_worker"],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(app_root),
                env=env,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return exc
        except Exception as exc:
            return exc


def _encode_crop_png(crop_bgr: np.ndarray) -> str:
    crop_rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
    image = Image.fromarray(crop_rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _worker_error(proc: subprocess.CompletedProcess) -> str:
    detail = (proc.stderr or proc.stdout or "").strip()
    code = f"worker exit code {proc.returncode}"
    return f"{code}: {detail}" if detail else code


def _empty_analysis(error: str, class_name: str = "") -> dict:
    return {
        "class_name": class_name,
        "top_match": {},
        "matches": [],
        "possible_causes": [],
        "inspection_advice": [],
        "error": error,
    }


_ANALYZER: CauseAnalyzer | None = None
_ANALYZER_LOCK = threading.Lock()


def get_cause_analyzer() -> CauseAnalyzer:
    global _ANALYZER
    with _ANALYZER_LOCK:
        if _ANALYZER is None:
            _ANALYZER = CauseAnalyzer()
        return _ANALYZER
