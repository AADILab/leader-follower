import functools
from typing import Dict, List
import numpy as np
from gym.spaces import Box
import enum

from boids_manager import BoidsManager


class OBSERVATION(enum.IntEnum):
    GOAL_AND_CENTROID = 0

class REWARD(enum.IntEnum):
    DISTANCE_TO_GOAL = 0

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

    def getRewards(self, bm: BoidsManager, actions: Dict, step_count: int, total_steps: int, posisble_agents: List[str]):
        """Get the rewards for each agent based on the simulation state and the agents' actions."""
        # For now, I'm going to return a list of team-wide rewards with no agent-based breakdown
        # This function takes in possible_agents so I can assign agent-based rewards later if I want to

        # Calculate temporal weight
        wt = 1 - np.cos(np.pi * step_count/total_steps)

        # Calculate average distance with temporal weight
        reward_list = []
        follower_positions = bm.get_follower_positions()
        for goal_location in self.goal_locations:
            # Calculate distance between the goal location and all followers
            distances = np.linalg.norm(goal_location - follower_positions)

            # Weight these distances by the temporal weight and overall steps
            weighted_distances = distances * wt / total_steps

            # Average the distances for this timestep
            avg_distance = np.average(weighted_distances)

            # Append the average distance for this particular goal location
            reward_list.append(avg_distance)

        return {"team": reward_list}

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
