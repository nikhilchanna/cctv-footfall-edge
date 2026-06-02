import logging
import os
from pathlib import Path

from cv_engine.config.schema import (
    DetectorSchema,
    EngineSchema,
    FootfallSchema,
    InferenceSchema,
    OcclusionSchema,
    TrackerSchema,
    ZoneSchema,
)
from cv_engine.counting.zone_generator import ZoneConfigError, ZoneGenerator, ZonePolygons
from cv_engine.model_pool import ModelPool

logger = logging.getLogger(__name__)


def _resolve_inference_device(requested: str) -> str:
    import torch

    device = requested or "cpu"
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            return device
        logger.warning("CUDA not available — using cpu")
        return "cpu"
    if device == "mps":
        logger.warning("mps requested but YOLO uses cpu on this stack")
        return "cpu"
    return device


DEFAULTS = {
    "detector.model_path": "models/scut_head_yolov8n.pt",
    "detector.detector_type": "head_scut",
    "detector.class_ids": [0],
    "detector.conf_threshold": 0.30,
    "detector.iou_threshold": 0.45,
    "detector.min_bbox_pixels": 6,
    "detector.max_aspect_ratio": 4.0,
    "tracker.type": "bytetrack",
    "tracker.track_thresh": 0.5,
    "tracker.track_buffer": 30,
    "zones.auto_generate": True,
    "zones.observation_offset_pixels": 150,
    "zones.count_zone_width_pixels": 100,
    "zones.ignore_offset_pixels": 100,
    "zones.entry_side": "above",
    "occlusion.lost_track_timeout_seconds": 5.0,
    "occlusion.reattach_threshold_px": 80.0,
    "footfall.camera_role": "IN",
    "footfall.count_direction": "both",
    "inference.device": os.getenv("CV_ENGINE_DEVICE", "cuda:0"),
    "inference.max_concurrent": int(os.getenv("CV_ENGINE_MAX_CONCURRENT", "2")),
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _defaults_dict() -> dict:
    d: dict = {}
    for key, val in DEFAULTS.items():
        parts = key.split(".")
        cur = d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return d


def _extract_points(zone_block: dict | None) -> list[tuple[float, float]] | None:
    if not zone_block:
        return None
    pts = zone_block.get("points")
    if not pts or len(pts) < 3:
        return None
    return [(float(p[0]), float(p[1])) for p in pts]


def _zone_int(zones_raw: dict, new_key: str, legacy_key: str, default: int) -> int:
    if new_key in zones_raw:
        return int(zones_raw[new_key])
    if legacy_key in zones_raw:
        return int(zones_raw[legacy_key])
    return default


def _resolve_manual_points(zones_raw: dict) -> tuple[
    list[tuple[float, float]] | None,
    list[tuple[float, float]] | None,
    list[tuple[float, float]] | None,
]:
    obs = _extract_points(zones_raw.get("observation")) or _extract_points(zones_raw.get("entry"))
    cnt = _extract_points(zones_raw.get("count")) or _extract_points(zones_raw.get("buffer"))
    ign = _extract_points(zones_raw.get("ignore")) or _extract_points(zones_raw.get("exit"))
    return obs, cnt, ign


def resolve_zone_polygons(
    zones_cfg: ZoneSchema,
    line_coords: dict | None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> ZonePolygons:
    if zones_cfg.observation_points and zones_cfg.count_points and zones_cfg.ignore_points:
        return ZonePolygons(
            observation=zones_cfg.observation_points,
            count=zones_cfg.count_points,
            ignore=zones_cfg.ignore_points,
        )
    if (
        not zones_cfg.auto_generate
        and (zones_cfg.observation_points or zones_cfg.count_points or zones_cfg.ignore_points)
    ):
        logger.warning("Incomplete manual zones — falling back to auto_generate from line_coords")
    if line_coords:
        return ZoneGenerator.generate_from_line(
            line_coords,
            zones_cfg.observation_offset_pixels,
            zones_cfg.count_zone_width_pixels,
            zones_cfg.ignore_offset_pixels,
            frame_width=frame_width,
            frame_height=frame_height,
            entry_side=zones_cfg.entry_side,
        )
    raise ZoneConfigError("no zone config: need manual polygons or line_coords + auto_generate")


def _normalize_camera_role(raw: str) -> str:
    r = (raw or "IN").lower()
    if r in ("in", "entry", "footfall"):
        return "IN"
    if r in ("out", "exit"):
        return "OUT"
    if r == "occupancy_only":
        return "occupancy_only"
    return raw.upper()


def load_engine_config(
    cv_engine: dict | None,
    line_coords: dict | None = None,
    cv_engine_defaults: dict | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> EngineSchema:
    merged = _defaults_dict()
    if cv_engine_defaults:
        merged = _deep_merge(merged, cv_engine_defaults)
    if cv_engine:
        merged = _deep_merge(merged, cv_engine)

    det_raw = merged.get("detector", {})
    if "head" in det_raw and "model_path" not in det_raw:
        det_raw = det_raw.get("head", {})
    elif "person" in det_raw and "head" in det_raw:
        det_raw = det_raw.get("head", det_raw)

    model_path = det_raw.get("model_path", DEFAULTS["detector.model_path"])
    if not Path(model_path).exists() and Path("yolov8n.pt").exists():
        logger.warning("Head model missing at %s — using ./yolov8n.pt for dev", model_path)
        model_path = "yolov8n.pt"

    detector = DetectorSchema(
        model_path=model_path,
        detector_type=str(det_raw.get("detector_type", DEFAULTS["detector.detector_type"])),
        conf_threshold=float(det_raw.get("conf_threshold", DEFAULTS["detector.conf_threshold"])),
        iou_threshold=float(det_raw.get("iou_threshold", DEFAULTS["detector.iou_threshold"])),
        class_ids=list(det_raw.get("class_ids", DEFAULTS["detector.class_ids"])),
        min_bbox_pixels=int(det_raw.get("min_bbox_pixels", DEFAULTS["detector.min_bbox_pixels"])),
        max_aspect_ratio=float(det_raw.get("max_aspect_ratio", DEFAULTS["detector.max_aspect_ratio"])),
    )

    zones_raw = merged.get("zones", {})
    obs_pts, cnt_pts, ign_pts = _resolve_manual_points(zones_raw)
    zones = ZoneSchema(
        auto_generate=bool(zones_raw.get("auto_generate", DEFAULTS["zones.auto_generate"])),
        observation_offset_pixels=_zone_int(
            zones_raw, "observation_offset_pixels", "entry_offset_pixels", 150
        ),
        count_zone_width_pixels=_zone_int(
            zones_raw, "count_zone_width_pixels", "buffer_width_pixels", 100
        ),
        ignore_offset_pixels=_zone_int(
            zones_raw, "ignore_offset_pixels", "exit_offset_pixels", 100
        ),
        entry_side=str(zones_raw.get("entry_side", DEFAULTS["zones.entry_side"])),
        observation_points=obs_pts,
        count_points=cnt_pts,
        ignore_points=ign_pts,
    )

    occ_raw = merged.get("occlusion", {})
    occlusion = OcclusionSchema(
        lost_track_timeout_seconds=float(
            occ_raw.get(
                "lost_track_timeout_seconds",
                DEFAULTS["occlusion.lost_track_timeout_seconds"],
            )
        ),
        reattach_threshold_px=float(
            occ_raw.get("reattach_threshold_px", DEFAULTS["occlusion.reattach_threshold_px"])
        ),
    )
    if "max_lost_frames" in occ_raw and "lost_track_timeout_seconds" not in occ_raw:
        mlf = float(occ_raw["max_lost_frames"])
        if mlf <= 30:
            occlusion.lost_track_timeout_seconds = mlf / 7.0

    footfall_raw = merged.get("footfall", {})
    footfall = FootfallSchema(
        camera_role=_normalize_camera_role(
            str(footfall_raw.get("camera_role", DEFAULTS["footfall.camera_role"]))
        ),
        count_direction=str(
            footfall_raw.get("count_direction", DEFAULTS["footfall.count_direction"])
        ),
    )

    inference = InferenceSchema(
        device=_resolve_inference_device(str(merged.get("inference", {}).get("device", DEFAULTS["inference.device"]))),
        max_concurrent=int(merged.get("inference", {}).get("max_concurrent", DEFAULTS["inference.max_concurrent"])),
    )
    ModelPool.configure(inference.device, inference.max_concurrent)

    zone_polygons = resolve_zone_polygons(zones, line_coords, frame_width, frame_height)

    return EngineSchema(
        detector=detector,
        tracker=TrackerSchema(
            type=str(merged.get("tracker", {}).get("type", DEFAULTS["tracker.type"])),
            track_thresh=float(merged.get("tracker", {}).get("track_thresh", DEFAULTS["tracker.track_thresh"])),
            track_buffer=int(merged.get("tracker", {}).get("track_buffer", DEFAULTS["tracker.track_buffer"])),
        ),
        zones=zones,
        occlusion=occlusion,
        footfall=footfall,
        inference=inference,
        line_coords=line_coords,
        zone_polygons=zone_polygons,
    )


EngineConfig = EngineSchema
ZoneConfig = ZoneSchema
OcclusionConfig = OcclusionSchema
TrackerConfig = TrackerSchema

# Legacy modules still import these
from dataclasses import dataclass


@dataclass
class DensityConfig:
    low_threshold: float = 0.002
    high_threshold: float = 0.008


@dataclass
class ConfidenceConfig:
    det_weight: float = 0.4
    track_weight: float = 0.35
    density_weight: float = 0.25
