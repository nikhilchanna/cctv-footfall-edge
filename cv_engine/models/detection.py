from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]  # xyxy
    confidence: float
    class_id: int = 0
    detector_type: str = ""

    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])
