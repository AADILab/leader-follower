"""
@title

@description

"""
import argparse
from pathlib import Path

from leader_follower import project_properties
from leader_follower.agent import Poi, Follower, Leader
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork
from leader_follower.utils import load_config


def test_global(env):
    return

def test_difference(env):
    return

def test_dpp(env):
    return

def main(main_args):
    render_mode = 'rgb_array'
    delta_time = 1

    leader_obs_rad = 5
    repulsion_rad = 2
    attraction_rad = 5

    config_fn = Path(project_properties.config_dir, 'test.yaml')
    experiment_config = load_config(str(config_fn))

    # agent_id, policy_population: list[NeuralNetwork], location, velocity, sensor_resolution, observation_radius, value
    leaders = [
        Leader(idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, value=1,
               observation_radius=leader_obs_rad, policy=NeuralNetwork(n_inputs=8, n_hidden=2, n_outputs=2))
        for idx, each_pos in enumerate(experiment_config['leader_positions'])
    ]
    # agent_id, update_rule, location, velocity, sensor_resolution, observation_radius, value
    followers = [
        Follower(agent_id=idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, value=1,
                 repulsion_radius=repulsion_rad, repulsion_strength=2,
                 attraction_radius=attraction_rad, attraction_strength=1)
        for idx, each_pos in enumerate(experiment_config['follower_positions'])
    ]
    #  agent_id, location, velocity, sensor_resolution, observation_radius, value, coupling
    pois = [
        Poi(idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, value=1,
            observation_radius=leader_obs_rad, coupling=1)
        for idx, each_pos in enumerate(experiment_config['poi_positions'])
    ]

    # leaders: list[Leader], followers: list[Follower], pois: list[Poi], max_steps, delta_time=1, render_mode=None
    env = LeaderFollowerEnv(
        leaders=leaders, followers=followers, pois=pois, max_steps=100, render_mode=render_mode, delta_time=delta_time
    )

    test_global(env)
    test_difference(env)
    test_dpp(env)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
