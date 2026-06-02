from shapely.geometry import Point, Polygon

from cv_engine.counting.zone_generator import ZonePolygons

ZONE_OBSERVATION = "observation"
ZONE_COUNT = "count"
ZONE_IGNORE = "ignore"


class ZoneManager:
    def __init__(self, zones: ZonePolygons):
        self._zones = zones
        self._polys = {
            ZONE_OBSERVATION: Polygon(zones.observation),
            ZONE_COUNT: Polygon(zones.count),
            ZONE_IGNORE: Polygon(zones.ignore),
        }

    def zone_at(self, centroid: tuple[float, float]) -> str | None:
        pt = Point(centroid[0], centroid[1])
        # Far → near order. Observation before count — shared edge belongs to observation.
        if self._polys[ZONE_OBSERVATION].contains(pt) or self._polys[ZONE_OBSERVATION].touches(pt):
            if not self._polys[ZONE_COUNT].contains(pt):
                return ZONE_OBSERVATION
        if self._polys[ZONE_COUNT].contains(pt) or self._polys[ZONE_COUNT].touches(pt):
            return ZONE_COUNT
        if self._polys[ZONE_IGNORE].contains(pt) or self._polys[ZONE_IGNORE].touches(pt):
            return ZONE_IGNORE
        return None

    def get_overlay(self) -> dict:
        return {
            "observation": self._zones.observation,
            "count": self._zones.count,
            "ignore": self._zones.ignore,
            # Legacy keys for UI + cv_processor overlay
            "entry": self._zones.observation,
            "buffer": self._zones.count,
            "exit": self._zones.ignore,
        }

    @property
    def polygons(self) -> ZonePolygons:
        return self._zones
