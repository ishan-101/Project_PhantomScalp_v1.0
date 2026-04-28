from fastapi import FastAPI
from .onnx_infer import ONNXModel

app = FastAPI()
model = ONNXModel("data/store/model.onnx")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    feats = payload.get("features", [])
    p = model.predict_proba(feats)
    return {"proba_long": p, "proba_short": 1 - p}