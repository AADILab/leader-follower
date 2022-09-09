import numpy as np
from numpy.typing import NDArray

class POI():
    def __init__(self, position: np.array) -> None:
        self.position = position
        self.observed = False
        # Groups of agents that observed this POI
        self.observation_list = []

class POIColony():
    def __init__(self, positions: NDArray[np.float64], observation_radius: float, coupling: int) -> None:
        self.positions = positions
        self.pois = [POI(position) for position in positions]
        self.num_pois = len(self.pois)
        self.observation_radius = observation_radius
        self.coupling = coupling

    def numObserved(self):
        return sum(poi.observed for poi in self.pois)

    def reset(self) -> None:
        for poi in self.pois:
            poi.observed = False
