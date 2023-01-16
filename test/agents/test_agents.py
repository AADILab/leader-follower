"""
@title

@description

"""
import argparse
from pathlib import Path

from leader_follower import project_properties
from leader_follower.agent import Follower, Leader, Poi
from leader_follower.learn.neural_network import NeuralNetwork
from leader_follower.utils import load_config


def test_leader(positions):
    leaders = [
        Leader(idx, location=(1, 1), velocity=(0, 0), sensor_resolution=4, observation_radius=1, value=1,
               policy_population=[NeuralNetwork(8, 2, 2)])
        for idx, each_pos in enumerate(positions)
    ]
    return


def test_follower(positions):
    followers = [
        Follower(agent_id=idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, observation_radius=1, value=1,
                 repulsion_radius=0.25, repulsion_strength=2, attraction_radius=2, attraction_strength=1)
        for idx, each_pos in enumerate(positions)
    ]
    return


def test_poi(positions):
    pois = [
        Poi(idx, location=(1, 9), velocity=(0, 0), sensor_resolution=4, observation_radius=1, value=1, coupling=1)
        for idx, each_pos in enumerate(positions)
    ]
    return


def main(main_args):
    config_fn = Path(project_properties.config_dir, 'test.yaml')
    experiment_config = load_config(str(config_fn))

    # todo agent tests
    test_leader(experiment_config['leader_positions'])
    test_follower(experiment_config['follower_positions'])
    test_poi(experiment_config['poi_positions'])
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
