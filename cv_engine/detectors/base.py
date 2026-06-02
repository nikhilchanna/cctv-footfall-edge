from abc import ABC, abstractmethod

import numpy as np

from cv_engine.models.detection import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...

    @property
    @abstractmethod
    def detector_type(self) -> str:
        ...
