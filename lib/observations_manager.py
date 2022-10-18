from enum import IntEnum
from typing import Optional, List, Union

import numpy as np
from numpy.typing import NDArray
from gym.spaces import Box

from lib.math_helpers import calculateCentroid, calculateDistance, boundAnglePiToPi
from lib.boids_colony import BoidsColony, Boid
from lib.poi_colony import POIColony, POI

class ObservationRule(IntEnum):
    Individual = 0

class SensorType(IntEnum):
    InverseDistanceCentroid = 0
    Density = 1

class ObservationManager():
    def __init__(self,
        observation_rule: Union[ObservationRule, str],
        boids_colony: BoidsColony,
        poi_colony: POIColony,
        map_dimensions: Optional[List[float]],
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
            self.observation_radius = np.sqrt(np.max(map_dimensions)**2 + np.max(map_dimensions)**2)
        else:
            self.observation_radius = observation_radius

        self.num_poi_bins = num_poi_bins
        self.num_swarm_bins = num_swarm_bins

        self.poi_sensor_type = poi_sensor_type
        self.swarm_sensor_type = swarm_sensor_type

    def getObservationSpace(self):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces

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

        lows =  [low_poi for _ in range(self.num_poi_bins)]  + [low_swarm for _ in range(self.num_swarm_bins)]
        highs = [high_poi for _ in range(self.num_poi_bins)] + [high_swarm for _ in range(self.num_swarm_bins)]

        return Box(
            low=np.array(lows, dtype=np.float64),
            high=np.array(highs, dtype=np.float64),
            dtype=np.float64
        )

    def getSensorReading(self, bin: List[Union[Boid,POI]], boid: Boid, sensor_type: SensorType) -> float:
        if sensor_type.value == SensorType.InverseDistanceCentroid:
            if len(bin) == 0:
                return 0.0
            else:
                # Get all the positions in this quadrant
                relative_positions = np.array([item.position - boid.position for item in bin])
                # Calculate the centroid of the positions
                center_position = calculateCentroid(relative_positions)
                # Distance to centroid
            return (self.observation_radius - np.linalg.norm(center_position))/self.observation_radius
        elif sensor_type.value == SensorType.Density:
            if len(bin) == 0:
                return 0.0
            else:
                # Get all the positions in this quadrant
                relative_positions = np.array([item.position - boid.position for item in bin])
                # Calculate distance to all these positions
                distances = np.linalg.norm(relative_positions, axis=1)
                # Turn that into density
                return np.sum([1-distance/self.observation_radius for distance in distances])

    def getSensorReadings(self, bins: List[List[Union[Boid,POI]]], boid: Boid, sensor_type: SensorType) -> float:
        return np.array([self.getSensorReading(bin, boid, sensor_type) for bin in bins])

    def generateBins(self, boid: Boid, num_bins: int, items: List[Union[Boid,POI]]):
        bins = [[] for _ in range(num_bins)]
        bin_size = 2*np.pi / num_bins
        for item in items:
            relative_position = item.position - boid.position
            distance = np.linalg.norm(relative_position)
            if distance <= self.observation_radius:
                # Bin POI according to angle into correct bin
                # Angle is in world frame from boid to poi
                angle = np.arctan2(relative_position[1], relative_position[0])
                # Relative angle is angle relative to boid heading
                relative_angle = boundAnglePiToPi(angle - boid.heading)
                # Technically, these angles are the same. This makes binning easier, though
                if relative_angle == np.pi: relative_angle = -np.pi
                # Determine which bin this position belongs to
                bin_number = int( (relative_angle+np.pi)/bin_size )
                # Bin the position properly
                bins[bin_number].append(item)
        return bins

    def generatePoiBins(self, boid: Boid):
        return self.generateBins(boid, self.num_poi_bins, self.poi_colony.pois)

    def getPoiSensorReadings(self, bins: List[List[POI]], boid: Boid):
        if self.poi_sensor_type == SensorType.Density:
            return self.getSensorReadings(bins, boid, self.poi_sensor_type)/float(self.poi_colony.num_pois)
        return self.getSensorReadings(bins, boid, self.poi_sensor_type)

    def getPoiObservation(self, boid: Boid):
        # Bin observable pois into bins
        bins = self.generatePoiBins(boid)
        # Turn pois into bins into one sensor reading per bin
        poi_readings = self.getPoiSensorReadings(bins, boid)
        return np.array(poi_readings, dtype=np.float64)

    def generateSwarmBins(self, boid: Boid):
        return self.generateBins(boid, self.num_swarm_bins, self.boids_colony.getObservableBoids(boid))

    def getSwarmSensorReadings(self, bins: List[List[Boid]], boid: Boid):
        if self.swarm_sensor_type == SensorType.Density:
            return self.getSensorReadings(bins, boid, self.swarm_sensor_type)/float(self.boids_colony.bounds.num_total)
        return self.getSensorReadings(bins, boid, self.swarm_sensor_type)

    def getSwarmObservation(self, boid: Boid):
        # Bin observable boids into bins
        bins = self.generateSwarmBins(boid)
        # Turn bins into one sensor reading per bin
        swarm_readings = self.getSwarmSensorReadings(bins, boid)
        return np.array(swarm_readings, dtype=np.float64)

    def getObservation(self, boid: Boid) -> NDArray[np.float64]:
        poi_observation = self.getPoiObservation(boid)
        swarm_observation = self.getSwarmObservation(boid)
        return np.hstack((poi_observation, swarm_observation))

    def getAllObservations(self):
        observations = []
        for leader in self.boids_colony.getLeaders():
            observations.append(self.getObservation(leader))
        return observations

    # def getObservationS
