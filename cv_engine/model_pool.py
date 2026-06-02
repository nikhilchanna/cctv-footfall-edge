import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)


class ModelPool:
    _models: dict = {}
    _lock = threading.Lock()
    _semaphore: threading.Semaphore | None = None
    _device: str = "cuda:0"
    _configured = False

    @classmethod
    def configure(cls, device: str, max_concurrent: int) -> None:
        with cls._lock:
            cls._device = device
            cls._semaphore = threading.Semaphore(max(1, max_concurrent))
            cls._configured = True

    @classmethod
    def _ensure_loaded(cls, model_path: str):
        from ultralytics import YOLO

        if model_path in cls._models:
            return cls._models[model_path]

        logger.info("Loading model: %s on %s", model_path, cls._device)
        model = YOLO(model_path)
        # First predict fuses batchnorm — must run once before any concurrent infer
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        model(dummy, device=cls._device, verbose=False)
        cls._models[model_path] = model
        logger.info("Model ready: %s", model_path)
        return model

    @classmethod
    def infer(cls, model_path: str, frame, **kwargs):
        if cls._semaphore is None:
            cls.configure(
                os.getenv("CV_ENGINE_DEVICE", "cuda:0"),
                int(os.getenv("CV_ENGINE_MAX_CONCURRENT", "2")),
            )
        with cls._semaphore:
            with cls._lock:
                model = cls._ensure_loaded(model_path)
                return model(frame, device=cls._device, **kwargs)
