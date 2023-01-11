from enum import IntEnum
from functools import partial

import numpy as np

from leader_follower.utils import calc_centroid, bound_angle


class ObservationRule(IntEnum):
    Individual = 0


class SensorType(IntEnum):
    InverseDistanceCentroid = 0
    Density = 1


def density_sensory_reading(normalize, boid, sensor_bin, observation_radius):
    # Get all the positions in this quadrant
    # Calculate distance to all these positions
    # Turn that into density
    relative_positions = np.array([item.position - boid.position for item in sensor_bin])
    distances = np.linalg.norm(relative_positions, axis=1)
    return np.sum([1 - distance / observation_radius for distance in distances]) / normalize


def inversion_sensor_reading(normalize, boid, sensor_bin, observation_radius):
    # Get all the positions in this quadrant
    # Calculate the centroid of the positions
    # Distance to centroid
    relative_positions = np.array([item.position - boid.position for item in sensor_bin])
    center_position = calc_centroid(relative_positions)
    return (observation_radius - np.linalg.norm(center_position)) / (observation_radius * normalize)


def generate_bins(boid, num_bins, items, observation_radius):
    # todo optimize
    bins = [[] for _ in range(num_bins)]
    bin_size = 2 * np.pi / num_bins
    for item in items:
        relative_position = item.position - boid.position
        distance = np.linalg.norm(relative_position)
        if distance <= observation_radius:
            # Bin POI according to angle into correct bin
            # Angle is in world frame from boid to poi
            angle = np.arctan2(relative_position[1], relative_position[0])
            # Relative angle is angle relative to boid heading
            relative_angle = bound_angle(angle - boid.heading)
            # Technically, these angles are the same. This makes binning easier, though
            if relative_angle == np.pi: relative_angle = -np.pi
            # Determine which bin this position belongs to
            bin_number = int((relative_angle + np.pi) / bin_size)
            # Bin the position properly
            bins[bin_number].append(item)
    return bins


def get_item_observation(boid, num_bins, observable_items, num_items, sensor_type, observation_radius):
    # Bin observable boid/item into bins
    # Turn bins into one sensor reading per bin
    bins = generate_bins(boid, num_bins, observable_items, observation_radius)

    normalize_factor = float(num_items) if sensor_type == SensorType.Density else 1
    sensor_map = {
        SensorType.InverseDistanceCentroid: partial(inversion_sensor_reading, normalize_factor),
        SensorType.Density: partial(density_sensory_reading, normalize_factor)
    }
    sensor_func = sensor_map[sensor_type.value]

    sensor_readings = np.array([
        sensor_func(boid, each_bin, observation_radius) if len(each_bin) != 0 else 0.0
        for each_bin in bins
    ])
    return np.array(sensor_readings, dtype=np.float64)


def get_observation(boid, num_poi_bins, num_swarm_bins, poi_colony, boids_colony, sensor_type, observation_radius):
    poi_observation = get_item_observation(
        boid, num_poi_bins, poi_colony.pois, float(poi_colony.num_pois), sensor_type, observation_radius
    )
    swarm_observation = get_item_observation(
        boid, num_swarm_bins, boids_colony.getObservableBoids(boid), boids_colony.bounds.num_total,
        sensor_type, observation_radius
    )
    return np.hstack((poi_observation, swarm_observation))


def get_all_observations(num_poi_bins, num_swarm_bins, poi_colony, boids_colony, sensor_type, obs_radius):
    observations = [
        get_observation(leader, num_poi_bins, num_swarm_bins, poi_colony, boids_colony, sensor_type, obs_radius)
        for leader in boids_colony.getLeaders()
    ]
    return observations
