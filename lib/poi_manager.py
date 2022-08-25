from typing import List
import numpy as np

from lib.boids_manager import BoidsManager

class POI():
    def __init__(self, position: np.array, observation_radius: float) -> None:
        self.position = position
        self.observed = False
        self.observation_radius = observation_radius

class POIManager():
    def __init__(self, positions: List[List[float]], observation_radius: float, coupling: int = 1) -> None:
        self.positions = np.array(positions)
        self.pois = [POI(position, observation_radius) for position in positions]
        self.num_pois = len(self.pois)
        self.coupling = coupling
        self.observation_radius = observation_radius

    def numObserved(self):
        return sum(poi.observed for poi in self.pois)

    def update(self, bm: BoidsManager) -> None:
        for poi in self.pois:
            # print(np.linalg.norm(bm.positions - poi.position, axis=1))
            distances = np.linalg.norm(bm.positions - poi.position, axis=1)
            observations = np.sum(distances<=self.observation_radius)
            if observations >= self.coupling:
                poi.observed = True
                # print("POI Observed | min distance: ", min_distance, " | observation_radius: ", self.observation_radius)

    def reset(self) -> None:
        for poi in self.pois:
            poi.observed = False
