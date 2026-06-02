from abc import ABC, abstractmethod

import numpy as np

from cv_engine.models.detection import Detection
from cv_engine.models.track import Track


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
