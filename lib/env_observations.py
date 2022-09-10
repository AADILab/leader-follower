from enum import IntEnum
from typing import Optional, List, Union

import numpy as np
from numpy.typing import NDArray

from lib.math_helpers import calculateCentroid, calculateDistance, boundAnglePiToPi
from lib.boids_colony import BoidsColony, Boid
from lib.poi_colony import POIColony, POI

class ObservationRule(IntEnum):
    Individual = 0

class SensorType(IntEnum):
    InverseDistance = 0

class ObservationManager():
    def __init__(self,
        observation_rule: ObservationRule,
        boids_colony: BoidsColony,
        poi_colony: POIColony,
        num_poi_bins: Optional[int],
        num_swarm_bins: Optional[int],
        poi_sensor_type: Optional[SensorType],
        swarm_sensor_type: Optional[SensorType]
        ) -> None:
        self.observation_rule = observation_rule
        self.boids_colony = boids_colony
        self.poi_colony = poi_colony

        self.num_poi_bins = num_poi_bins
        self.num_swarm_bins = num_swarm_bins

        self.poi_sensor_type = poi_sensor_type
        self.swarm_sensor_type = swarm_sensor_type

    def getSensorReading(self, bin: List[Union[Boid,POI]], boid: Boid, sensor_type: SensorType) -> float:
        if sensor_type.value == SensorType.InverseDistance:
            if len(bin) == 0:
                return 0.0
            else:
                # Get all the positions in this quadrant
                relative_positions = np.array([item.position - boid.position for item in bin])
                # Calculate the centroid of the positions
                center_position = calculateCentroid(relative_positions)
                # Distance to centroid
            return self.boids_colony.radius_attraction - np.linalg.norm(center_position)

    def getSensorReadings(self, bins: List[List[Union[Boid,POI]]], boid: Boid, sensor_type: SensorType) -> float:
        return [self.getSensorReading(bin, boid, sensor_type) for bin in bins]

    def generateBins(self, boid: Boid, num_bins: int, items: List[Union[Boid,POI]]):
        bins = [[] for _ in range(num_bins)]
        bin_size = 2*np.pi / num_bins
        for item in items:
            relative_position = item.position - boid.position
            distance = np.linalg.norm(relative_position)
            if distance <= self.boids_colony.radius_attraction:
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
