from enum import IntEnum
from typing import Optional, List, Union

import numpy as np
from gym.spaces import Box
from numpy.typing import NDArray

from leader_follower.agents.boids_colony import BoidsColony, Boid
from leader_follower.math_helpers import calc_centroid, bound_angle_pi_pi
from leader_follower.agents.poi_colony import POIColony, POI


class ObservationRule(IntEnum):
    Individual = 0


class SensorType(IntEnum):
    InverseDistanceCentroid = 0
    Density = 1


class ObservationManager:
    def __init__(self,
                 observation_rule: Union[ObservationRule, str],
                 boids_colony: BoidsColony,
                 poi_colony: POIColony,
                 map_dimensions,
                 observation_radius: Optional[int],
                 num_poi_bins: Optional[int],
                 num_swarm_bins: Optional[int],
                 poi_sensor_type: Optional[Union[SensorType, str]],
                 swarm_sensor_type: Optional[Union[SensorType, str]],
                 full_observability: bool = False
                 ) -> None:

        if type(observation_rule) == str:
            observation_rule = ObservationRule[observation_rule]
        if type(poi_sensor_type) == str:
            poi_sensor_type = SensorType[poi_sensor_type]
        if type(swarm_sensor_type) == str:
            swarm_sensor_type = SensorType[swarm_sensor_type]

        self.observation_rule = observation_rule
        self.boids_colony = boids_colony
        self.poi_colony = poi_colony

        if full_observability:
            self.observation_radius = np.sqrt(np.max(map_dimensions) ** 2 + np.max(map_dimensions) ** 2)
        else:
            self.observation_radius = observation_radius

        self.num_poi_bins = num_poi_bins
        self.num_swarm_bins = num_swarm_bins

        self.poi_sensor_type = poi_sensor_type
        self.swarm_sensor_type = swarm_sensor_type
        return

    def get_observation_space(self):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        low_poi = 0
        high_poi = 1
        low_swarm = 0
        high_swarm = 1

        if self.poi_sensor_type.value == SensorType.InverseDistanceCentroid:
            low_poi = 0
            high_poi = self.observation_radius

        elif self.poi_sensor_type.value == SensorType.Density:
            low_poi = 0
            high_poi = 1.0

        if self.swarm_sensor_type.value == SensorType.InverseDistanceCentroid:
            low_swarm = 0
            high_swarm = self.observation_radius

        elif self.poi_sensor_type.value == SensorType.Density:
            low_swarm = 0
            high_swarm = 1.0

        lows = [low_poi for _ in range(self.num_poi_bins)] + [low_swarm for _ in range(self.num_swarm_bins)]
        highs = [high_poi for _ in range(self.num_poi_bins)] + [high_swarm for _ in range(self.num_swarm_bins)]

        return Box(
            low=np.array(lows, dtype=np.float64),
            high=np.array(highs, dtype=np.float64),
            dtype=np.float64
        )

    def get_sensor_reading(self, sensor_bin: List[Union[Boid, POI]], boid: Boid, sensor_type: SensorType):
        if sensor_type.value == SensorType.InverseDistanceCentroid:
            if len(sensor_bin) == 0:
                return 0.0
            else:
                # Get all the positions in this quadrant
                relative_positions = np.array([item.position - boid.position for item in sensor_bin])
                # Calculate the centroid of the positions
                center_position = calc_centroid(relative_positions)
                # Distance to centroid
            return (self.observation_radius - np.linalg.norm(center_position)) / self.observation_radius
        elif sensor_type.value == SensorType.Density:
            if len(sensor_bin) == 0:
                return 0.0
            else:
                # Get all the positions in this quadrant
                relative_positions = np.array([item.position - boid.position for item in sensor_bin])
                # Calculate distance to all these positions
                distances = np.linalg.norm(relative_positions, axis=1)
                # Turn that into density
                return np.sum([1 - distance / self.observation_radius for distance in distances])

    def get_sensor_readings(self, bins: List[List[Union[Boid, POI]]], boid: Boid, sensor_type: SensorType):
        return np.array([self.get_sensor_reading(sensor_bin, boid, sensor_type) for sensor_bin in bins])

    def generate_bins(self, boid: Boid, num_bins: int, items: List[Union[Boid, POI]]):
        bins = [[] for _ in range(num_bins)]
        bin_size = 2 * np.pi / num_bins
        for item in items:
            relative_position = item.position - boid.position
            distance = np.linalg.norm(relative_position)
            if distance <= self.observation_radius:
                # Bin POI according to angle into correct bin
                # Angle is in world frame from boid to poi
                angle = np.arctan2(relative_position[1], relative_position[0])
                # Relative angle is angle relative to boid heading
                relative_angle = bound_angle_pi_pi(angle - boid.heading)
                # Technically, these angles are the same. This makes binning easier, though
                if relative_angle == np.pi:
                    relative_angle = -np.pi
                # Determine which bin this position belongs to
                bin_number = int((relative_angle + np.pi) / bin_size)
                # Bin the position properly
                bins[bin_number].append(item)
        return bins

    def generate_poi_bins(self, boid: Boid):
        return self.generate_bins(boid, self.num_poi_bins, self.poi_colony.pois)

    def get_poi_sensor_readings(self, bins: List[List[POI]], boid: Boid):
        if self.poi_sensor_type == SensorType.Density:
            return self.get_sensor_readings(bins, boid, self.poi_sensor_type) / float(self.poi_colony.num_pois)
        return self.get_sensor_readings(bins, boid, self.poi_sensor_type)

    def get_poi_observation(self, boid: Boid):
        # Bin observable pois into bins
        bins = self.generate_poi_bins(boid)
        # Turn pois into bins into one sensor reading per bin
        poi_readings = self.get_poi_sensor_readings(bins, boid)
        return np.array(poi_readings, dtype=np.float64)

    def generate_swarm_bins(self, boid: Boid):
        return self.generate_bins(boid, self.num_swarm_bins, self.boids_colony.observable_boids(boid))

    def get_swarm_sensor_readings(self, bins: List[List[Boid]], boid: Boid):
        if self.swarm_sensor_type == SensorType.Density:
            return self.get_sensor_readings(bins, boid, self.swarm_sensor_type) / float(
                self.boids_colony.bounds.num_total)
        return self.get_sensor_readings(bins, boid, self.swarm_sensor_type)

    def get_swarm_observation(self, boid: Boid):
        # Bin observable boids into bins
        bins = self.generate_swarm_bins(boid)
        # Turn bins into one sensor reading per bin
        swarm_readings = self.get_swarm_sensor_readings(bins, boid)
        return np.array(swarm_readings, dtype=np.float64)

    def get_observation(self, boid: Boid) -> NDArray[np.float64]:
        poi_observation = self.get_poi_observation(boid)
        swarm_observation = self.get_swarm_observation(boid)
        return np.hstack((poi_observation, swarm_observation))

    def get_all_observations(self):
        observations = []
        for leader in self.boids_colony.leaders():
            observations.append(self.get_observation(leader))
        return observations
