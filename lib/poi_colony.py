import numpy as np
from numpy.typing import NDArray

from lib.math_helpers import calculateDistance
from lib.colony_helpers import BoidsColonyState

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

    def updatePois(self, boids_colony_state: BoidsColonyState):
        for poi in self.pois:
            distances = calculateDistance(poi.position, boids_colony_state.positions)
            num_observations = np.sum(distances<=self.observation_radius)
            if num_observations >= self.coupling:
                poi.observed = True
                # Get ids of swarm members that observed this poi
                observer_ids = np.nonzero(distances<=self.observation_radius)[0]
                poi.observation_list.append(observer_ids)

    def numObserved(self):
        return sum(poi.observed for poi in self.pois)

    def reset(self) -> None:
        for poi in self.pois:
            poi.observed = False
