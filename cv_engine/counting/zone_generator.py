import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ZoneConfigError(ValueError):
    pass


@dataclass
class ZonePolygons:
    observation: list[tuple[float, float]]
    count: list[tuple[float, float]]
    ignore: list[tuple[float, float]]

    # UI + legacy overlay keys
    @property
    def entry(self) -> list[tuple[float, float]]:
        return self.observation

    @property
    def buffer(self) -> list[tuple[float, float]]:
        return self.count

    @property
    def exit(self) -> list[tuple[float, float]]:
        return self.ignore


class ZoneGenerator:
    @staticmethod
    def generate_from_line(
        line_coords: dict,
        observation_offset: int,
        count_zone_width: int,
        ignore_offset: int,
        frame_width: int | None = None,
        frame_height: int | None = None,
        entry_side: str = "above",
    ) -> ZonePolygons:
        if not line_coords:
            raise ZoneConfigError("line_coords missing")
        x1 = float(line_coords.get("x1", 0))
        y1 = float(line_coords.get("y1", 0))
        x2 = float(line_coords.get("x2", 0))
        y2 = float(line_coords.get("y2", 0))
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            raise ZoneConfigError("line has zero length")

        ux, uy = dx / length, dy / length
        # above = far side (observation), below = camera side (ignore)
        if entry_side == "above":
            nx, ny = uy, -ux
        else:
            nx, ny = -uy, ux

        def _shift(px, py, dist):
            return (px + nx * dist, py + ny * dist)

        def _clamp_point(px, py):
            if frame_width is not None:
                px = max(0.0, min(float(frame_width), px))
            if frame_height is not None:
                py = max(0.0, min(float(frame_height), py))
            return (px, py)

        def _poly(p1, p2, d1, d2):
            pts = [
                _shift(p1[0], p1[1], d1),
                _shift(p2[0], p2[1], d1),
                _shift(p2[0], p2[1], d2),
                _shift(p1[0], p1[1], d2),
            ]
            return [_clamp_point(x, y) for x, y in pts]

        p1 = (x1, y1)
        p2 = (x2, y2)
        half_count = count_zone_width / 2.0

        # Non-overlapping bands: far road → observation → count → ignore → camera
        observation = _poly(p1, p2, half_count, observation_offset)
        count_zone = _poly(p1, p2, -half_count, half_count)
        ignore = _poly(p1, p2, -ignore_offset, -half_count)

        return ZonePolygons(observation=observation, count=count_zone, ignore=ignore)
