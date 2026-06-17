"""
CLIP/SigLIP cause analysis wrapper.

The FastAPI/YOLO process imports OpenCV. On this Windows environment, loading
the SigLIP torch model in the same process after cv2 can trigger a native access
violation. To keep the server stable, SigLIP runs in a persistent worker
subprocess that does not import cv2 and reuses the loaded model.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_NAME = "google/siglip-so400m-patch14-384"
PROMPTS_PATH = Path(__file__).with_name("cause_prompts.json")
WORKER_TIMEOUT_SEC = int(os.getenv("CAUSE_WORKER_TIMEOUT_SEC", "240"))
CPU_FALLBACK_TIMEOUT_SEC = int(os.getenv("CAUSE_CPU_FALLBACK_TIMEOUT_SEC", "600"))
WINDOWS_ACCESS_VIOLATION_EXIT = 3221225477


class _WorkerCrashed(RuntimeError):
    def __init__(self, returncode: int | None):
        super().__init__(f"cause worker exited with code {returncode}")
        self.returncode = returncode


class _WorkerClient:
    def __init__(self, app_root: Path, env: dict):
        self._lock = threading.Lock()
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "app.cause_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(app_root),
            env=env,
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _read_stdout(self):
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._responses.put(line)
        self._proc.wait()
        self._responses.put(None)

    def request(self, payload: dict, timeout_sec: int) -> dict:
        with self._lock:
            if self._proc.poll() is not None:
                raise _WorkerCrashed(self._proc.returncode)
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                raise _WorkerCrashed(self._proc.poll())
            try:
                response = self._responses.get(timeout=timeout_sec)
            except queue.Empty:
                self.close()
                raise TimeoutError
            if response is None:
                raise _WorkerCrashed(self._proc.poll())
            return json.loads(response)

    def close(self):
        if self._proc.poll() is None:
            self._proc.kill()


@dataclass
class CauseAnalyzer:
    model_name: str = MODEL_NAME

    def __post_init__(self):
        self._prompt_library = None
        self._prompt_error: str | None = None
        self._prompt_lock = threading.Lock()
        self._workers: dict[str, _WorkerClient] = {}
        self._workers_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._warmed_up = False
        atexit.register(self.close)

    def close(self):
        with self._workers_lock:
            for worker in self._workers.values():
                worker.close()
            self._workers.clear()

    def warmup(self) -> dict:
        with self._warmup_lock:
            if self._warmed_up:
                return {"status": "ready", "model": self.model_name, "warmup_time_ms": 0}

            self._load_prompts()
            if not self._prompt_library:
                raise RuntimeError(self._prompt_error or "prompt library unavailable")

            started = time.perf_counter()
            result = self.analyze_crop(
                np.zeros((384, 384, 3), dtype=np.uint8),
                next(iter(self._prompt_library)),
                top_k=1,
            )
            if result.get("error"):
                raise RuntimeError(result["error"])
            self._warmed_up = True
            return {
                "status": "ready",
                "model": self.model_name,
                "warmup_time_ms": round((time.perf_counter() - started) * 1000),
            }

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

        try:
            data = self._request_worker(payload, app_root, env, WORKER_TIMEOUT_SEC)
        except TimeoutError:
            return [
                _empty_analysis(f"cause worker timed out after {WORKER_TIMEOUT_SEC}s")
                for _ in payload_items
            ]
        except _WorkerCrashed as exc:
            if (
                exc.returncode in {WINDOWS_ACCESS_VIOLATION_EXIT, -1073741819}
                and env.get("CAUSE_MODEL_DEVICE", "auto").lower() != "cpu"
            ):
                cpu_env = env.copy()
                cpu_env["CAUSE_MODEL_DEVICE"] = "cpu"
                try:
                    data = self._request_worker(
                        payload, app_root, cpu_env, CPU_FALLBACK_TIMEOUT_SEC
                    )
                except Exception as cpu_exc:
                    return [
                        _empty_analysis(f"GPU worker crashed; CPU fallback failed: {cpu_exc}")
                        for _ in payload_items
                    ]
            else:
                return [_empty_analysis(str(exc)) for _ in payload_items]
        except Exception as exc:
            return [
                _empty_analysis(f"cause worker failed: {exc}") for _ in payload_items
            ]
        if data.get("error"):
            err = f"cause worker failed: {data['error']}"
            return [_empty_analysis(err) for _ in payload_items]
        return data.get("results", [])

    def _request_worker(
        self,
        payload: dict,
        app_root: Path,
        env: dict,
        timeout_sec: int,
    ) -> dict:
        device = env.get("CAUSE_MODEL_DEVICE", "auto").lower()
        with self._workers_lock:
            worker = self._workers.get(device)
            if worker is None or worker._proc.poll() is not None:
                worker = _WorkerClient(app_root, env)
                self._workers[device] = worker
        try:
            return worker.request(payload, timeout_sec)
        except (TimeoutError, _WorkerCrashed):
            with self._workers_lock:
                if self._workers.get(device) is worker:
                    worker.close()
                    del self._workers[device]
            raise


def _encode_crop_png(crop_bgr: np.ndarray) -> str:
    crop_rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
    image = Image.fromarray(crop_rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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
