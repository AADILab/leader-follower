from typing import List, Optional
import numpy as np

from lib.colony_helpers import ColonyState, StateBounds

class BoidSpawner():
    def __init__(self,
        bounds: StateBounds


        # position_rule: int,
        # velocity_rule: int,
        # heading_rule: int,
        # # For varying spawn
        # num_leaders: int = None,
        # num_followers: int = None,
        # # Fixed initial spawn
        # leader_positions: List[List[float]] = None, follower_positions: List[List[float]] = None,
        # leader_headings: List[float] = None, follower_headings: List[float] = None,
        # leader_velocities: List[float] = None, follower_velocities: List[float] = None
        ) -> None:
        self.bounds = bounds
        pass

        # self.position_rule = position_rule
        # self.velocity_rule = velocity_rule
        # self.heading_rule = heading_rule

        # self.leader_positions = leader_positions
        # self.follower_positions = follower_positions
        # self.leader_headings = leader_headings
        # self.follower_headings = follower_headings
        # self.leader_velocities = leader_velocities
        # self.follower_velocities = follower_velocities

    def generateSpawnState(self):
        positions = np.hstack((
                np.random.uniform(0, self.bounds.map_dimensions[0], size=(self.bounds.num_total,1)),
                np.random.uniform(0, self.bounds.map_dimensions[1], size=(self.bounds.num_total,1))
            ))
        velocities = np.random.uniform(self.bounds.min_velocity, self.bounds.max_velocity, size=self.bounds.num_total)
        headings = np.random.uniform(0, 2*np.pi, size=self.bounds.num_total)
        is_leader = np.array( [True for _ in range(self.bounds.num_leaders)] + [False for _ in range(self.bounds.num_followers)] )
        return ColonyState(positions, velocities, headings, is_leader)




