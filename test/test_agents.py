"""
@title

@description

"""
import argparse
import inspect
import itertools

import numpy as np

from leader_follower import project_properties
from leader_follower.agent import Follower, Leader, Poi
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork


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

def print_test_separator():
    caller_func = inspect.stack()[1][3]
    print(f'{"=" * project_properties.TERMINAL_COLUMNS}')
    print(f'{"=" * project_properties.TERMINAL_COLUMNS}')
    print(f'{"=" * project_properties.TERMINAL_COLUMNS}')
    print(f'Running {caller_func} tests')
    return


def test_leader_multiple(num_surround):
    print_test_separator()
    sensor_resolution = 4
    leader_obs_rad = 5
    leader_value = 1
    test_network = NeuralNetwork(n_inputs=sensor_resolution * 2, n_outputs = 2)

    follower_value = 1
    repulsion_radius = 1
    repulsion_strength = 5
    attraction_radius = 5
    attraction_strength = 1

    poi_value = 0
    poi_obs_rad = 1
    poi_coupling = 1

    base_loc = (0, 0)
    surround_locs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    base_network = NeuralNetwork(n_inputs=sensor_resolution * 2, n_outputs = 2)
    base_leader = Leader(
        0, location=base_loc, sensor_resolution=sensor_resolution, observation_radius=leader_obs_rad,
        value=leader_value, policy=base_network
    )

    surround_permutations = list(itertools.permutations(surround_locs, r=num_surround))
    for idx, locs in enumerate(surround_permutations):
        leads = [base_leader]
        for loc_idx, each_loc in enumerate(locs):
            each_leader = Leader(
                loc_idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=leader_obs_rad,
                value=leader_value, policy=test_network
            )
            leads.append(each_leader)
        env = LeaderFollowerEnv(leaders=leads, followers=[], pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, locs in enumerate(surround_permutations):
        leads = [base_leader]
        followers = []
        for loc_idx, each_loc in enumerate(locs):
            each_follower = Follower(
                loc_idx + 1, location=each_loc, sensor_resolution=sensor_resolution,value=follower_value,
                repulsion_radius=repulsion_radius, repulsion_strength=repulsion_strength,
                attraction_radius=attraction_radius, attraction_strength=attraction_strength
            )
            followers.append(each_follower)
        env = LeaderFollowerEnv(
            leaders=leads, followers=followers, pois=[], max_steps=100, render_mode=None, delta_time=1
        )
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, locs in enumerate(surround_permutations):
        leads = [base_leader]
        pois = []
        for loc_idx, each_loc in enumerate(locs):
            each_poi = Poi(
                loc_idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=poi_obs_rad,
                value=poi_value, coupling=poi_coupling
            )
            pois.append(each_poi)
        env = LeaderFollowerEnv(leaders=leads, followers=[], pois=pois, max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)
    return


def test_follower_multiple(num_surround):
    print_test_separator()
    sensor_resolution = 4
    leader_obs_rad = 5
    leader_value = 1
    test_network = NeuralNetwork(n_inputs=sensor_resolution * 2, n_outputs=2)

    follower_value = 1
    repulsion_radius = 1
    repulsion_strength = 5
    attraction_radius = 5
    attraction_strength = 1

    poi_value = 0
    poi_obs_rad = 1
    poi_coupling = 1

    base_loc = (0, 0)
    surround_locs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    base_follower = Follower(
        0, location=base_loc, sensor_resolution=sensor_resolution, value=follower_value,
        repulsion_radius=repulsion_radius, repulsion_strength=repulsion_strength,
        attraction_radius=attraction_radius, attraction_strength=attraction_strength
    )

    surround_permutations = list(itertools.permutations(surround_locs, r=num_surround))
    for idx, locs in enumerate(surround_permutations):
        followers = [base_follower]
        leads = []
        for loc_idx, each_loc in enumerate(locs):
            each_leader = Leader(
                loc_idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=leader_obs_rad,
                value=leader_value, policy=test_network
            )
            leads.append(each_leader)
        env = LeaderFollowerEnv(leaders=leads, followers=followers, pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, locs in enumerate(surround_permutations):
        followers = []
        for loc_idx, each_loc in enumerate(locs):
            each_follower = Follower(
                loc_idx + 1, location=each_loc, sensor_resolution=sensor_resolution, value=follower_value,
                repulsion_radius=repulsion_radius, repulsion_strength=repulsion_strength,
                attraction_radius=attraction_radius, attraction_strength=attraction_strength
            )
            followers.append(each_follower)
        env = LeaderFollowerEnv(
            leaders=[], followers=followers, pois=[], max_steps=100, render_mode=None, delta_time=1
        )
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, locs in enumerate(surround_permutations):
        followers = [base_follower]
        pois = []
        for loc_idx, each_loc in enumerate(locs):
            each_poi = Poi(
                loc_idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=poi_obs_rad,
                value=poi_value, coupling=poi_coupling
            )
            pois.append(each_poi)
        env = LeaderFollowerEnv(leaders=[], followers=followers, pois=pois, max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)
    return


def test_poi_multiple(num_surround):
    print_test_separator()
    sensor_resolution = 4
    leader_obs_rad = 5
    leader_value = 1
    test_network = NeuralNetwork(n_inputs=sensor_resolution * 2, n_outputs=2)

    follower_value = 1
    repulsion_radius = 1
    repulsion_strength = 5
    attraction_radius = 5
    attraction_strength = 1

    poi_value = 0
    poi_obs_rad = 1
    poi_coupling = 1

    base_loc = (0, 0)
    surround_locs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    base_poi = Poi(0, location=base_loc, sensor_resolution=sensor_resolution, observation_radius=poi_obs_rad, value=poi_value, coupling=poi_coupling)

    surround_permutations = list(itertools.permutations(surround_locs, r=num_surround))
    for idx, locs in enumerate(surround_permutations):
        pois = [base_poi]
        leads = []
        for loc_idx, each_loc in enumerate(locs):
            each_leader = Leader(
                loc_idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=leader_obs_rad,
                value=leader_value, policy=test_network
            )
            leads.append(each_leader)
        env = LeaderFollowerEnv(leaders=leads, followers=[], pois=pois, max_steps=100, render_mode=None,
                                delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, locs in enumerate(surround_permutations):
        pois = [base_poi]
        followers = []
        for loc_idx, each_loc in enumerate(locs):
            each_follower = Follower(
                loc_idx + 1, location=each_loc, sensor_resolution=sensor_resolution, value=follower_value,
                repulsion_radius=repulsion_radius, repulsion_strength=repulsion_strength,
                attraction_radius=attraction_radius, attraction_strength=attraction_strength
            )
            followers.append(each_follower)
        env = LeaderFollowerEnv(
            leaders=[], followers=followers, pois=pois, max_steps=100, render_mode=None, delta_time=1
        )
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, locs in enumerate(surround_permutations):
        pois = [base_poi]
        for loc_idx, each_loc in enumerate(locs):
            each_poi = Poi(
                loc_idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=poi_obs_rad,
                value=poi_value, coupling=poi_coupling
            )
            pois.append(each_poi)
        env = LeaderFollowerEnv(leaders=[], followers=[], pois=pois, max_steps=100, render_mode=None,
                                delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)
    return

def main(main_args):
    max_agents = 4
    for num_agents in range(1, max_agents):
        test_leader_multiple(num_agents)
        test_follower_multiple(num_agents)
        test_poi_multiple(num_agents)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
