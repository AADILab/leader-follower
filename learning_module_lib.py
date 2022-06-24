import functools
from typing import Dict, List
import numpy as np
from gym.spaces import Box

from boids_manager import BoidsManager

class LearningModule():
    """This class will contain methods for getting the observations and rewards for leader agents.
    This module will contain all information related to the swarm objectives.
    """
    def __init__(self, goal_locations: np.ndarray) -> None:
        self.goal_locations = goal_locations
        self.num_goals = goal_locations.shape[0]

    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces

        # [distance to goal, theta to goal, distance to swarm centroid, theta to swarm centroid]
        # All thetas are from agent's own reference frame. -pi to pi
        return Box(
            low=np.array([-np.inf, -np.pi, -np.inf, -np.pi]),
            high=np.array([np.inf, np.pi, np.inf, np.pi]),
            dtype=np.float32)

    def getRewards(self, bm: BoidsManager, actions: Dict):
        """Get the rewards for each agent based on the simulation state and the agents' actions."""
        pass

    def getObservations(self, bm: BoidsManager):
        """Get the complete observations for each agent."""
        # Get the centroid of observable boids (leaders+followers) for each leader agent
        centroids_obs_np = bm.get_leader_centroid_observations()
        # Get the distances and angles of leader agents to the goal locations
        all_goal_obs = bm.get_leader_relative_position_observations(self.goal_locations)

        observations = {}
        for agent_id in range(bm.num_leaders):
            observations[agent_id] = np.hstack((centroids_obs_np[agent_id], all_goal_obs[agent_id].flatten()))
        return observations
