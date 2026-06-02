from cv_engine.config.schema import EngineSchema
from cv_engine.detectors.base import Detector
from cv_engine.detectors.head_detector import HeadDetector
from cv_engine.model_pool import ModelPool

SUPPORTED_HEAD_TYPES = {"head_scut", "head_crowdhuman", "scut_head", "crowdhuman_head"}


def build_head_detector(cfg: EngineSchema) -> Detector:
    ModelPool.configure(cfg.inference.device, cfg.inference.max_concurrent)
    det = cfg.detector
    dtype = det.detector_type.lower()
    if dtype not in SUPPORTED_HEAD_TYPES and not dtype.startswith("head_"):
        raise ValueError(f"unsupported head detector type: {det.detector_type}")
    return HeadDetector(
        model_path=det.model_path,
        detector_type=det.detector_type,
        class_ids=list(det.class_ids),
        conf_threshold=det.conf_threshold,
        iou_threshold=det.iou_threshold,
        min_bbox_pixels=det.min_bbox_pixels,
        max_aspect_ratio=det.max_aspect_ratio,
    )


def build_detectors(cfg: EngineSchema) -> dict[str, Detector | None]:
    """Legacy API for video test engine — head model used for both keys."""
    head = build_head_detector(cfg)
    return {"person": head, "head": head}
