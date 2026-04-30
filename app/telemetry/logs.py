# app/telemetry/logs.py
import logging

def get_logger(level: str = "INFO"):
    logger = logging.getLogger("phantomscalp")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
