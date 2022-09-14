from enum import IntEnum
from typing import Optional, List, Union

import numpy as np
from numpy.typing import NDArray

from lib.math_helpers import randomPositions


class POISpawnRule(IntEnum):
    Set = 0 # Set poi positions manually
    UniformRandom = 1 # Randomly distribute pois accross entire map
    BoundedRandom = 2 # Randomly distribute pois bounded by a certain limit
    # EdgeRandom = 2 # Randomly place POIs on only near the edges of the map (Possible future implementation)
    BoundedCircle = 3 # Randomly place POIs within two concentric circular bands

class POISpawner():
    def __init__(self,
        poi_spawn_rule: Union[POISpawnRule, str],
        # number of POIs for generated spawns
        num_pois: Optional[int] = None,
        # Bounds
        bound_fraction: Optional[float] = None,
        inner_circle_bound_fraction: Optional[float] = None,
        outer_circle_bound_fraction: Optional[float] = None,
        # Whether to fix spawns
        fix_poi_spawns: bool = False,
        # Map dimensions for generated spawns
        map_dimensions: Optional[NDArray[np.float64]] = None,
        edge_min_fraction: Optional[float] = None,
        # Positions for preset spawns
        positions: Optional[List[List[float]]] = None
        ) -> None:

        if type(poi_spawn_rule) == str:
            poi_spawn_rule = POISpawnRule[poi_spawn_rule]

        self.spawn_rule = poi_spawn_rule
        self.num_pois = num_pois
        self.bound_fraction = bound_fraction
        self.inner_circle_bound_fraction = inner_circle_bound_fraction
        self.outer_circle_bound_fraction = outer_circle_bound_fraction
        self.fix_poi_spawns = fix_poi_spawns
        self.map_dimensions = map_dimensions
        self.edge_min_fraction = edge_min_fraction
        self.positions = positions

        if self.fix_poi_spawns:
            self.fixed_positions = self.generatePositions()

    def generatePositions(self) -> NDArray[np.float64]:
        if self.spawn_rule.value == POISpawnRule.Set.value:
            return np.array(self.positions)
        elif self.spawn_rule.value == POISpawnRule.UniformRandom.value:
            return randomPositions(self.map_dimensions, self.num_pois)
        elif self.spawn_rule.value == POISpawnRule.BoundedRandom.value:
            low_bounds = self.map_dimensions/2 * (1-self.bound_fraction)
            high_bounds = self.map_dimensions/2 * self.bound_fraction + self.map_dimensions/2
            return randomPositions(low_bounds=low_bounds, high_bounds=high_bounds, num_positions=self.num_pois)
        elif self.spawn_rule.value == POISpawnRule.BoundedCircle.value:
            return self.generateBoundedCircle()

    def generateBoundedCircle(self):
        # Generate a bunch of radii from 0 to bound (outer - inner)
        outer_bound = np.min(self.map_dimensions)/2 * self.outer_circle_bound_fraction
        inner_bound = np.min(self.map_dimensions)/2 * self.inner_circle_bound_fraction
        threshold = outer_bound - inner_bound
        radii = inner_bound + np.random.uniform(0, threshold, size=(self.num_pois,1))
        thetas = np.random.uniform(0, 2*np.pi, size=(self.num_pois,1))
        return self.map_dimensions/2 +  np.hstack((
            radii*np.cos(thetas),
            radii*np.sin(thetas)
        ))

    def getSpawnPositions(self):
        if self.fix_poi_spawns:
            return self.fixed_positions.copy()
        else:
            return self.generatePositions()
