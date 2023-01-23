"""
@title

@description

"""
import argparse
from pathlib import Path

import numpy as np

from leader_follower import project_properties
from leader_follower.agent import Follower, Leader, Poi
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork
from leader_follower.positions import load_positions

def print_environment_step(env, obs, acts):
    print(f'{"=" * project_properties.TERMINAL_COLUMNS}')
    with np.printoptions(precision=3, suppress=True):
        for a_name in env.agents:
            agent = env.agent_mapping[a_name]
            a_loc = agent.location
            a_act = acts[a_name]
            a_obs = obs[a_name]

            if isinstance(a_obs, np.ndarray):
                a_obs = np.array_str(a_obs)
            if isinstance(a_act, np.ndarray):
                a_act = np.array_str(a_act)

            print(f'{a_name=}')
            print(f'\t{a_loc=}')
            print(f'\t{a_obs=}')
            print(f'\t{a_act=}')
    print(f'{"=" * project_properties.TERMINAL_COLUMNS}')
    return

def test_leader():
    sensor_resolution = 4

    base_network = NeuralNetwork(n_inputs=sensor_resolution * 2, n_outputs = 2)
    test_network = NeuralNetwork(n_inputs=sensor_resolution * 2, n_outputs = 2)

    base_loc = (5, 5)
    base_leader = Leader(0, location=base_loc, sensor_resolution=sensor_resolution, observation_radius=5, value=1,
                         policy=base_network)
    # todo look into why value in observation is slightly smaller when agent is at the boundary of a quadrant
    # todo why is observation twice the size of the sensor resolution
    surround_locs = [(5, 4), (5, 6), (4, 5), (6, 5), (4, 4), (4, 6), (6, 4), (6, 6)]

    for idx, each_loc in enumerate(surround_locs):
        each_leader = Leader(idx + 1, location=each_loc, sensor_resolution=4, observation_radius=5, value=1,
                             policy=test_network)
        leads = [base_leader, each_leader]
        env = LeaderFollowerEnv(leaders=leads, followers=[], pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    configs_dir = Path(project_properties.test_config_dir, 'leaders')
    config_fns = configs_dir.glob('*.yaml')
    for con_fn in config_fns:
        load_positions(con_fn)
    return


def test_follower():
    sensor_resolution = 4

    repulsion_radius = 1
    repulsion_strength = 5

    attraction_radius = 5
    attraction_strength = 1

    base_loc = (5, 5)
    base_follower = Follower(
        0, location=base_loc, sensor_resolution=sensor_resolution, value=1,
        repulsion_radius=repulsion_radius, repulsion_strength=repulsion_strength,
        attraction_radius=attraction_radius, attraction_strength=attraction_strength
    )
    surround_locs = [(5, 4), (5, 6), (4, 5), (6, 5), (4, 4), (4, 6), (6, 4), (6, 6)]

    for idx, each_loc in enumerate(surround_locs):
        each_follower = Follower(idx + 1, location=each_loc,  sensor_resolution=sensor_resolution, value=1,
                                 repulsion_radius=repulsion_radius, repulsion_strength=repulsion_strength,
                                 attraction_radius=attraction_radius, attraction_strength=attraction_strength
                                 )
        followers = [base_follower, each_follower]
        env = LeaderFollowerEnv(leaders=[], followers=followers, pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)
    return


def test_poi():
    base_loc = (5, 5)
    base_poi = Poi(0, location=base_loc, sensor_resolution=4, observation_radius=1, value=1,coupling=1)
    surround_locs = [(5, 4), (5, 6), (4, 5), (6, 5), (4, 4), (4, 6), (6, 4), (6, 6)]

    for idx, each_loc in enumerate(surround_locs):
        each_poi = Poi(idx + 1, location=each_loc,  sensor_resolution=4, observation_radius=1, value=1, coupling=1)
        pois = [base_poi, each_poi]
        env = LeaderFollowerEnv(leaders=[], followers=[], pois=pois, max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)
    return


def main(main_args):
    test_leader()
    test_follower()
    test_poi()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
