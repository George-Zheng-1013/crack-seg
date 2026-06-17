"""Subprocess worker for SigLIP cause matching.

This module intentionally does not import cv2. It is executed with
`python -m app.cause_worker` by `cause_analyzer.py`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LOCAL_ONLY = os.getenv("CAUSE_MODEL_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
DEVICE_SETTING = os.getenv("CAUSE_MODEL_DEVICE", "auto").lower()


def main() -> int:
    _suppress_windows_crash_dialog()
    model_name = None
    prompt_path = None
    model_bundle = None
    prompt_library = None

    for line in sys.stdin:
        try:
            payload = json.loads(line)
            next_model_name = payload["model_name"]
            next_prompt_path = Path(payload["prompts_path"])
            if prompt_library is None or next_prompt_path != prompt_path:
                prompt_library = _load_prompt_library(next_prompt_path)
                prompt_path = next_prompt_path
            if model_bundle is None or next_model_name != model_name:
                model_bundle = _load_model(next_model_name)
                model_name = next_model_name

            top_k = int(payload.get("top_k", 3))
            results = [
                _analyze_item(item, prompt_library, model_bundle, top_k)
                for item in payload.get("items", [])
            ]
            print(json.dumps({"results": results}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
    return 0


def _suppress_windows_crash_dialog() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        SEM_NOOPENFILEERRORBOX = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            SEM_FAILCRITICALERRORS
            | SEM_NOGPFAULTERRORBOX
            | SEM_NOOPENFILEERRORBOX
        )
    except Exception:
        pass


def _load_prompt_library(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(class_name).lower(): records
        for class_name, records in raw.items()
        if isinstance(records, list)
    }


def _load_model(model_name: str) -> dict:
    import torch
    from transformers import AutoModel, SiglipProcessor

    print("SigLIP worker: selecting device", file=sys.stderr, flush=True)
    if DEVICE_SETTING == "cpu":
        device = "cpu"
    elif DEVICE_SETTING == "cuda":
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"SigLIP worker: loading processor on {device}", file=sys.stderr, flush=True)
    processor = SiglipProcessor.from_pretrained(
        model_name,
        local_files_only=LOCAL_ONLY,
    )
    print("SigLIP worker: loading model", file=sys.stderr, flush=True)
    try:
        model = AutoModel.from_pretrained(
            model_name,
            local_files_only=LOCAL_ONLY,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            model_name,
            local_files_only=LOCAL_ONLY,
            torch_dtype=dtype,
        )
    print("SigLIP worker: moving model to device", file=sys.stderr, flush=True)

    return {
        "processor": processor,
        "model": model.to(device).eval(),
        "device": device,
        "torch": torch,
    }


def _analyze_item(
    item: dict,
    prompt_library: dict,
    model_bundle: dict,
    top_k: int,
) -> dict:
    class_key = str(item.get("class_name") or "").lower()
    prompt_records = prompt_library.get(class_key, [])
    if not prompt_records:
        return _empty_analysis(f"no prompts for class: {class_key}", class_key)

    candidate_labels = [
        str(record["feature_prompt"])
        for record in prompt_records
        if record.get("feature_prompt")
    ]
    record_by_prompt = {
        str(record["feature_prompt"]): record
        for record in prompt_records
        if record.get("feature_prompt")
    }
    if not candidate_labels:
        return _empty_analysis(f"no feature prompts for class: {class_key}", class_key)

    try:
        pil_image = _decode_image(item["image_b64"])
        outputs = _predict(pil_image, candidate_labels, model_bundle)
        matches = [
            _match_from_output(output, record_by_prompt)
            for output in outputs[:top_k]
        ]
        top = matches[0] if matches else {}
        return {
            "class_name": class_key,
            "top_match": top,
            "matches": matches,
            "possible_causes": [top["cause"]] if top.get("cause") else [],
            "inspection_advice": top.get("inspection_advice", []),
            "note": "图文匹配结果仅用于辅助判断，报告中应表述为最可能成因或可能成因。",
        }
    except Exception as exc:
        return _empty_analysis(str(exc), class_key)


def _decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _predict(
    pil_image: Image.Image,
    candidate_labels: list[str],
    model_bundle: dict,
) -> list[dict]:
    processor = model_bundle["processor"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    torch = model_bundle["torch"]

    inputs = processor(
        text=candidate_labels,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        print("SigLIP worker: running inference", file=sys.stderr, flush=True)
        outputs = model(**inputs)

    scores = (
        outputs.logits_per_image[0]
        .softmax(dim=-1)
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    order = np.argsort(scores)[::-1]
    return [
        {"label": candidate_labels[int(idx)], "score": float(scores[int(idx)])}
        for idx in order
    ]


def _match_from_output(item: dict, record_by_prompt: dict) -> dict:
    prompt = item["label"]
    record = record_by_prompt.get(prompt, {})
    feature_cn = record.get("feature_cn", prompt)
    return {
        "cause": record.get("cause", ""),
        "feature_prompt": prompt,
        "feature_cn": feature_cn,
        "score": round(float(item["score"]), 4),
        "inspection_advice": record.get("inspection_advice", []),
        "label": prompt,
        "label_cn": feature_cn,
    }


def _empty_analysis(error: str, class_name: str = "") -> dict:
    return {
        "class_name": class_name,
        "top_match": {},
        "matches": [],
        "possible_causes": [],
        "inspection_advice": [],
        "error": error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
