"""
@title

@description

"""
import argparse
import inspect

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

def test_leader_single():
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
    base_leader = Leader(0, location=base_loc, sensor_resolution=sensor_resolution, observation_radius=leader_obs_rad, value=leader_value,
                         policy=base_network)

    for idx, each_loc in enumerate(surround_locs):
        each_leader = Leader(idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=leader_obs_rad, value=leader_value,
                             policy=test_network)
        leads = [base_leader, each_leader]
        env = LeaderFollowerEnv(leaders=leads, followers=[], pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, each_loc in enumerate(surround_locs):
        each_follower = Follower(idx + 1, location=each_loc, sensor_resolution=sensor_resolution, value=follower_value,
                                 repulsion_radius=repulsion_radius, repulsion_strength=repulsion_strength,
                                 attraction_radius=attraction_radius, attraction_strength=attraction_strength
                                 )
        followers = [each_follower]
        env = LeaderFollowerEnv(leaders=[base_leader], followers=followers, pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, each_loc in enumerate(surround_locs):
        each_poi = Poi(idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=poi_obs_rad, value=poi_value, coupling=poi_coupling)
        pois = [each_poi]
        env = LeaderFollowerEnv(leaders=[base_leader], followers=[], pois=pois, max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    # todo test rollout using the starting configurations in the configs/followers directory
    return

def test_leader_multiple(num_surround):
    # todo implement multiple agent tests for leader
    print_test_separator()
    return


def test_follower_single():
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

    for idx, each_loc in enumerate(surround_locs):
        each_leader = Leader(idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=leader_obs_rad, value=leader_value,
                             policy=test_network)
        leads = [each_leader]
        env = LeaderFollowerEnv(leaders=leads, followers=[base_follower], pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, each_loc in enumerate(surround_locs):
        each_follower = Follower(idx + 1, location=each_loc, sensor_resolution=sensor_resolution, value=follower_value,
                                 repulsion_radius=repulsion_radius, repulsion_strength=repulsion_strength,
                                 attraction_radius=attraction_radius, attraction_strength=attraction_strength
                                 )
        followers = [base_follower, each_follower]
        env = LeaderFollowerEnv(leaders=[], followers=followers, pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, each_loc in enumerate(surround_locs):
        each_poi = Poi(idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=poi_obs_rad, value=poi_value, coupling=poi_coupling)
        pois = [each_poi]
        env = LeaderFollowerEnv(leaders=[], followers=[base_follower], pois=pois, max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    # todo test rollout using the starting configurations in the configs/followers directory
    return

def test_follower_multiple(num_surround):
    # todo implement multiple agent tests for follower
    print_test_separator()
    return


def test_poi_single():
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

    for idx, each_loc in enumerate(surround_locs):
        each_leader = Leader(idx + 1, location=each_loc, sensor_resolution=sensor_resolution, observation_radius=leader_obs_rad, value=leader_value,
                             policy=test_network)
        leads = [each_leader]
        env = LeaderFollowerEnv(leaders=leads, followers=[], pois=[base_poi], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, each_loc in enumerate(surround_locs):
        each_follower = Follower(idx + 1, location=each_loc, sensor_resolution=sensor_resolution, value=follower_value,
                                 repulsion_radius=repulsion_radius, repulsion_strength=repulsion_strength,
                                 attraction_radius=attraction_radius, attraction_strength=attraction_strength
                                 )
        followers = [each_follower]
        env = LeaderFollowerEnv(leaders=[], followers=followers, pois=[base_poi], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    for idx, each_loc in enumerate(surround_locs):
        each_poi = Poi(idx + 1, location=each_loc,  sensor_resolution=sensor_resolution, observation_radius=poi_obs_rad, value=poi_value, coupling=poi_coupling)
        pois = [base_poi, each_poi]
        env = LeaderFollowerEnv(leaders=[], followers=[], pois=pois, max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print_environment_step(env, obs, acts)

    # todo test rollout using the starting configurations in the configs/pois directory
    return

def test_poi_multiple(num_surround):
    # todo implement multiple agent tests for leader
    print_test_separator()
    return

def main(main_args):
    test_leader_single()
    test_follower_single()
    test_poi_single()
    ###########################
    # todo loop up to 8 surrounding agents
    test_leader_multiple(2)
    test_follower_multiple(2)
    test_poi_multiple(2)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
