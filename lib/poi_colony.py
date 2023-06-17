from typing import Optional
import numpy as np
from numpy.typing import NDArray

from lib.math_helpers import calculateDistance
from lib.colony_helpers import BoidsColonyState
# from lib.fitness_calculator import FitnessCalculator, FolslowerSwitch

class POI():
    def __init__(self, positions: NDArray[np.float64], id: int) -> None:
        self._positions = positions
        self.id = id
        self.observed = False
        # Groups of agents that observed this POI
        self.observation_list = []

    @property
    def position(self) -> NDArray[np.float64]:
        return self._positions[self.id]

class POIColony():
    def __init__(self, positions: NDArray[np.float64], observation_radius: float, coupling: int) -> None:
        self.positions = positions
        self.pois = [POI(self.positions, id) for id in range(positions.shape[0])]
        self.num_pois = len(self.pois)
        self.observation_radius = observation_radius
        self.coupling = coupling

    def updatePois(self, boids_colony_state: BoidsColonyState):
        # Note: pretty sure this is deprecated because the fitness manager now tracks all of this stuff independently using the position history
        # or rather, the fitness manager just aggregates all the trajectories and does these type of calculations at the end
        pass
        # for poi in self.pois:
        #     if fitness_calculator.follower_switch == FollowerSwitch.UseLeadersAndFollowers:
        #         distances = calculateDistance(poi.position, boids_colony_state.positions)
        #     else:
        #         distances = calculateDistance(poi.position, boids_colony_state.positions[boids_colony_state.state_bounds.num_leaders:])
        #     num_observations = np.sum(distances<=self.observation_radius)
        #     if num_observations >= self.coupling:
        #         poi.observed = True
        #         # Get ids of swarm members that observed this poi
        #         observer_ids = np.nonzero(distances<=self.observation_radius)[0]
        #         poi.observation_list.append(observer_ids)

    def numObserved(self):
        return sum(poi.observed for poi in self.pois)

    def reset(self, positions: NDArray[np.float64], ) -> None:
        # Resetting self.positions should also reset position attribute for all pois
        self.positions[:,:] = positions
        for poi in self.pois:
            poi.observed = False
            poi.observation_list = []
