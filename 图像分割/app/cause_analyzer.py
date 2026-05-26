"""
CLIP/SigLIP 成因分析模块。

对裂缝裁剪图做零样本图文匹配，再将可见特征映射为可能成因。
该模块懒加载大模型，加载或推理失败时返回错误信息，不阻断主检测流程。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


MODEL_NAME = "google/siglip-so400m-patch14-384"

VISUAL_LABELS = [
    "thin concrete crack",
    "linear crack on concrete surface",
    "wide structural crack",
    "branching concrete crack",
    "spalling concrete surface",
    "peeling damaged concrete",
    "water seepage stain",
    "damp wall surface",
]

LABEL_TEXT_CN = {
    "thin concrete crack": "细小混凝土裂缝",
    "linear crack on concrete surface": "线性混凝土裂缝",
    "wide structural crack": "较宽结构性裂缝",
    "branching concrete crack": "分叉型混凝土裂缝",
    "spalling concrete surface": "混凝土表面剥落",
    "peeling damaged concrete": "表层脱落损伤",
    "water seepage stain": "渗水痕迹",
    "damp wall surface": "潮湿墙面",
}

CAUSE_RULES = {
    "thin concrete crack": {
        "possible_causes": ["干缩收缩", "温度应力", "表层老化"],
        "inspection_advice": ["复核裂缝宽度变化", "检查环境温湿度与养护记录"],
    },
    "linear crack on concrete surface": {
        "possible_causes": ["结构拉应力", "施工缝弱化", "温度变形"],
        "inspection_advice": ["结合构件受力方向复查", "观察裂缝是否持续扩展"],
    },
    "wide structural crack": {
        "possible_causes": ["承载不足", "沉降变形", "结构受力异常"],
        "inspection_advice": ["优先复核结构安全", "进行宽度监测和沉降观测"],
    },
    "branching concrete crack": {
        "possible_causes": ["疲劳损伤", "材料劣化", "冻融循环"],
        "inspection_advice": ["检查裂缝网络范围", "结合服役年限和环境暴露条件判断"],
    },
    "spalling concrete surface": {
        "possible_causes": ["钢筋锈蚀膨胀", "冻融破坏", "冲击损伤"],
        "inspection_advice": ["检查保护层厚度和钢筋锈蚀", "排查局部空鼓与脱落风险"],
    },
    "peeling damaged concrete": {
        "possible_causes": ["表层粘结失效", "长期风化", "施工质量缺陷"],
        "inspection_advice": ["检查表层强度", "清理松散区域后复查基层状态"],
    },
    "water seepage stain": {
        "possible_causes": ["防水层失效", "渗漏", "排水不良"],
        "inspection_advice": ["追踪水源路径", "检查排水和防水节点"],
    },
    "damp wall surface": {
        "possible_causes": ["长期潮湿", "渗水", "环境湿度过高"],
        "inspection_advice": ["测量局部含水率", "排查背水面和管线渗漏"],
    },
}


@dataclass
class CauseAnalyzer:
    model_name: str = MODEL_NAME

    def __post_init__(self):
        self._pipe = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    def _load(self):
        if self._pipe is not None or self._load_error is not None:
            return

        with self._lock:
            if self._pipe is not None or self._load_error is not None:
                return
            try:
                from transformers import pipeline

                self._pipe = pipeline(
                    task="zero-shot-image-classification",
                    model=self.model_name,
                )
            except Exception as exc:
                self._load_error = str(exc)

    def analyze_crop(self, crop_bgr: np.ndarray, top_k: int = 3) -> dict:
        if crop_bgr is None or crop_bgr.size == 0:
            return _empty_analysis("empty crop")

        self._load()
        if self._pipe is None:
            return _empty_analysis(self._load_error or "model unavailable")

        try:
            image_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            outputs = self._pipe(pil_image, candidate_labels=VISUAL_LABELS)
            matches = [
                {
                    "label": item["label"],
                    "label_cn": LABEL_TEXT_CN.get(item["label"], item["label"]),
                    "score": round(float(item["score"]), 4),
                }
                for item in outputs[:top_k]
            ]
            top = matches[0] if matches else {}
            rule = CAUSE_RULES.get(top.get("label"), {})
            return {
                "top_match": top,
                "matches": matches,
                "possible_causes": rule.get("possible_causes", []),
                "inspection_advice": rule.get("inspection_advice", []),
                "note": "图文匹配结果仅用于辅助判断，报告中应表述为可能成因。",
            }
        except Exception as exc:
            return _empty_analysis(str(exc))


def _empty_analysis(error: str) -> dict:
    return {
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
