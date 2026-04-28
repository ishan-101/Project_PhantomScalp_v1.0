from pydantic import BaseModel
from pathlib import Path
import os, yaml

class Paths(BaseModel):
    raw: Path
    clean: Path
    features: Path
    store: Path
    outputs: Path

class MLConf(BaseModel):
    target: str
    features: list[str]
    model: str
    onnx_path: Path

class ExecRisk(BaseModel):
    max_position: int
    daily_loss_cap_pct: float
    killswitch_drawdown_pct: float

class ExecConf(BaseModel):
    broker: str
    risk: ExecRisk

class TelemetryConf(BaseModel):
    slack_webhook: str | None
    log_level: str = "INFO"

class AppConf(BaseModel):
    mode: str
    timezone: str
    symbols: list[str]
    data_store: str

class Settings(BaseModel):
    app: AppConf
    paths: Paths
    ml: MLConf
    execution: ExecConf
    telemetry: TelemetryConf

def load_settings(path: str | os.PathLike) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)