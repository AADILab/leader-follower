"""
@title

@description

"""
from pathlib import Path

import numpy as np
from myaml import myaml
from scipy.ndimage.interpolation import rotate


##################
# Positions
##################


def random_positions(num_agents, lower_bound=0, upper_bound=1, seed=None):
    assert num_agents > 0
    assert lower_bound < upper_bound

    rng = np.random.default_rng(seed=seed)
    positions = rng.random((num_agents, 2))
    return positions

def concentric_positions(num_agents, map_dimensions, radii_fraction):
    positions_list = []
    for num_pois, fraction_radius in zip(num_agents, radii_fraction):
        radius = fraction_radius * min(map_dimensions) / 2
        thetas = np.expand_dims(np.linspace(0, 2 * np.pi, num_pois, endpoint=False), axis=1)
        positions = map_dimensions / 2 + np.hstack((
            radius * np.cos(thetas),
            radius * np.sin(thetas)
        ))
        positions_list.append(positions)
    return np.vstack(positions_list)


def circle_positions(num_agents, lower_bound=0, upper_bound=1):
    """
    square -> circle(num_agents=2)
    triangle -> circle(num_agents=3)
    square -> circle(num_agents=4)
    """
    assert num_agents > 0
    assert lower_bound < upper_bound
    angles = np.linspace(start=0, stop=2 * np.pi, num=num_agents, endpoint=False)
    positions = [
        [np.cos(each_angle), np.sin(each_angle)]
        for each_angle in angles
    ]
    positions = np.asarray(positions)
    return positions


def fixed_positions(num_agents, position):
    pos_vect = np.full(num_agents, position[0])
    for each_dim in position[1:]:
        next_vect = np.full(num_agents, each_dim)
        pos_vect = np.hstack((pos_vect, next_vect))
    return pos_vect


def parse_positions(config):
    leader_positions = config.get('leader_positions', 0)
    follower_positions = config.get('follower_positions', 0)
    return np.vstack((leader_positions, follower_positions))


def load_positions(positions_fname: Path):
    if not (positions_fname.exists() and positions_fname.is_file()):
        raise ValueError(f'Cannot find positions file:\n{positions_fname}')
    # read positions from file
    positions = myaml.safe_load(str(positions_fname))
    return positions


def linear_positions(num_agents, lower_bound=0, upper_bound=1):
    assert num_agents > 0
    assert lower_bound < upper_bound
    x_positions = np.linspace(start=(lower_bound,), stop=(upper_bound,), num=num_agents)
    y_positions = np.zeros((x_positions.size, 1))
    positions = np.hstack((x_positions, y_positions))
    return positions


##################
# Velocities
##################

def random_velocities(num_agents, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size=num_agents)


def parse_velocities(num_agents, min_velocity, max_velocity, config):
    velocity = config.get('velocity_fraction', 1) * (max_velocity - min_velocity) + min_velocity
    return velocity * np.ones(num_agents)


def fixed_velocities(num_agents, velocity):
    return np.full(num_agents, velocity)


##################
# Headings
##################

def random_headings(num_agents):
    return np.random.uniform(0, 2 * np.pi, size=num_agents)


def parse_headings(config):
    leader_headings = config.get('leader_headings', np.pi / 2)
    follower_headings = config.get('leader_headings', np.pi / 2)
    return np.hstack((leader_headings, follower_headings))


def fixed_headings(num_agents, heading):
    return np.full(num_agents, heading)


##################
# Transforms
##################

def scale_configuration(positions, scale: int | float | np.ndarray = 1):
    assert scale > 0
    assert len(positions) > 0
    if isinstance(scale, float) or isinstance(scale, int):
        scale = np.full(positions.shape[1], scale)
    for idx, each_scale in enumerate(scale):
        positions[:, idx] = each_scale * positions[:, idx]
    return positions


def rotate_configuration(positions, scale: int | float | np.ndarray = 1):
    assert len(positions) > 0
    rotated = rotate(positions, angle=scale)
    return rotated


def translate_configuration(positions, scale: int | float | np.ndarray = 1):
    assert scale > 0
    assert len(positions) > 0
    if isinstance(scale, float) or isinstance(scale, int):
        scale = np.full(positions.shape[1], scale)
    for idx, each_scale in enumerate(scale):
        positions[:, idx] = each_scale + positions[:, idx]
    return positions
