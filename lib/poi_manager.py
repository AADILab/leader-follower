from typing import List
import numpy as np

from lib.boids_manager import BoidsManager

class POI():
    def __init__(self, position: np.array) -> None:
        self.position = position
        self.observed = False

class POIManager():
    def __init__(self, positions: List[List[float]], observation_radius: float, coupling: int = 1) -> None:
        self.positions = np.array(positions)
        self.pois = [POI(position) for position in positions]
        self.num_pois = len(self.pois)
        self.coupling = coupling
        self.observation_radius = observation_radius

    def numObserved(self):
        return sum(poi.observed for poi in self.pois)

    def update(self, bm: BoidsManager) -> None:
        for poi in self.pois:
            # print(np.linalg.norm(bm.positions - poi.position, axis=1))
            min_distance = np.min(np.linalg.norm(bm.positions - poi.position, axis=1))
            if min_distance <= self.observation_radius:
                poi.observed = True
                # print("POI Observed | min distance: ", min_distance, " | observation_radius: ", self.observation_radius)

    def reset(self) -> None:
        for poi in self.pois:
            poi.observed = False
