import numpy as np
from lib.boids_manager import BoidsManager
from lib.poi_manager import POIManager

class ObservationManager():
    def __init__(self, bm: BoidsManager, pm: POIManager, include_followers: bool, invert_distances: bool=True) -> None:
        self.bm = bm
        self.pm = pm
        self.include_followers = include_followers
        self.invert_distances = invert_distances

    def getLeader4PoiDistances(self):
        """Get the observations of each leader for its distance to POIs within its observation radius divided into 4 quadrants
        Each quadrant is distance to poi in that quandrant. If there are several POI in one quadrant, closest distance is observed
        """
        all_poi_sensor_readings = np.zeros((self.bm.num_leaders,4))
        # Generate POI observation for each leader
        bin_size = 2*np.pi / 4
        for leader_id in range(self.bm.num_leaders):
            bins = [[] for _ in range(4)]
            # Figure out which POIs are within the observation radius
            for poi in self.pm.pois:
                # If poi is within observation radius, then save its position into the appropriate bin
                relative_position = poi.position - self.bm.positions[self.bm.num_followers+leader_id]
                distance_to_poi = np.linalg.norm(relative_position)
                if distance_to_poi < self.bm.radius_attraction:
                    # Bin POI according to angle into correct bins
                    angle = np.arctan2(relative_position[1], relative_position[0])
                    relative_angle = self.bm.bound_heading_pi_to_pi(angle - self.bm.headings[self.bm.num_followers+leader_id])
                    bin_num = int( (relative_angle+np.pi)/bin_size )
                    bins[bin_num].append(poi.position.copy())
            # Go back through each bin to get the actual sensor reading for each bin
            for bin_num, bin in enumerate(bins):
                if len(bin) == 0:
                    all_poi_sensor_readings[leader_id, bin_num] = self.bm.radius_attraction
                else:
                    # Calculate average position of that quadrant
                    average_position = self.bm.calculate_centroid(np.array(bins[bin_num]))
                    # Get relative position of leader to the poi average position
                    relative_position = average_position - self.bm.positions[leader_id+self.bm.num_followers]
                    # Get distance to that position
                    distance = np.sqrt(relative_position[0]**2 + relative_position[1]**2)
                    # Save the distance as the entry for that quadrant
                    all_poi_sensor_readings[leader_id, bin_num] = distance
            # Return all the poi sensor readings
            return all_poi_sensor_readings

    def getObservations(self):
        """Get the complete observations for each agent. 8 sensor readings for swarm. 4 sensor readings for pois"""
        # Swarm observations
        swarm_obs = self.bm.get_leader_8_swarm_obs()
        # Poi observations
        poi_obs = self.getLeader4PoiDistances()

        if self.invert_distances:
            swarm_obs = self.bm.radius_attraction - swarm_obs
            poi_obs = self.bm.radius_attraction - poi_obs

        observations = {}
        for agent_id in range(self.bm.num_leaders):
            if self.include_followers:
                observations[agent_id] = np.hstack((swarm_obs[agent_id], poi_obs[agent_id]))
            else:
                observations[agent_id] = poi_obs[agent_id]
        return observations

    def getObservationsOld(self):
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
