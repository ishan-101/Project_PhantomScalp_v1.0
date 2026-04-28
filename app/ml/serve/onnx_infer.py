# app/ml/serve/onnx_infer.py
from __future__ import annotations

import os
import json
import math

try:
    import numpy as np  # only needed if you use real ONNX
except Exception:
    np = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


class ONNXModel:
    """
    Unified inference wrapper:
    - If artifact ends with .onnx and onnxruntime is available -> use ORT
    - Else load a JSON linear model {"bias": float, "weights": {feat: w}} and sigmoid
    """
    def __init__(self, path: str):
        self.path = path
        self.mode = "json"
        _, ext = os.path.splitext(path)

        if ext.lower() == ".onnx" and ort is not None:
            # Real ONNX path
            self.mode = "onnx"
            self.sess = ort.InferenceSession(path)
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = self.sess.get_outputs()[0].name
        else:
            # JSON linear model (dummy v0.1)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.bias: float = float(obj.get("bias", 0.0))
            self.weights: dict[str, float] = {k: float(v) for k, v in obj.get("weights", {}).items()}
            # feature order we expect at predict time
            self.keys = ["rsi_14", "ema_20", "obv", "micro_price_imbalance"]

    def predict_proba(self, features) -> float:
        if self.mode == "onnx":
            if np is None:
                raise RuntimeError("numpy is required for ONNX inference but is not available.")
            x = np.array(features, dtype=np.float32).reshape(1, -1)
            proba = self.sess.run([self.output_name], {self.input_name: x})[0]
            # Be tolerant to different shapes: scalar / [ [p0, p1] ] / [ [p] ] / [p]
            try:
                arr = np.array(proba)
                if arr.ndim == 0:
                    return float(arr)
                flat = arr.flatten()
                # If we have 2 classes, return class-1 prob; otherwise return the single prob
                return float(flat[-1]) if flat.size > 1 else float(flat[0])
            except Exception:
                # last resort
                try:
                    return float(proba)
                except Exception as e:
                    raise RuntimeError(f"Unexpected ONNX output shape/type: {type(proba)}") from e
        else:
            # JSON linear model + sigmoid
            z = self.bias
            for k, x in zip(self.keys, features):
                z += float(self.weights.get(k, 0.0)) * float(x)
            return 1.0 / (1.0 + math.exp(-z))
