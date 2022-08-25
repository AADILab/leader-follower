import numpy as np
from lib.boids_manager import BoidsManager
from lib.poi_manager import POIManager

class ObservationManager():
    def __init__(self, bm: BoidsManager, pm: POIManager, include_followers) -> None:
        self.bm = bm
        self.pm = pm
        self.include_followers = include_followers

    def getObservations(self):
        """Get the complete observations for each agent."""
        # Get the centroid of observable boids (leaders+followers) for each leader agent
        centroids_obs_np = self.bm.get_leader_centroid_observations()
        # Get the distances and angles of leader agents to the goal locations
        all_goal_obs = self.bm.get_leader_relative_position_observations(self.pm.positions)

        observations = {}
        for agent_id in range(self.bm.num_leaders):
            if self.include_followers:
                observations[agent_id] = np.hstack((centroids_obs_np[agent_id], all_goal_obs[agent_id].flatten()))
            else:
                observations[agent_id] = all_goal_obs[agent_id].flatten()
        return observations
