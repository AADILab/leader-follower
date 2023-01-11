"""
@title

@description

"""
import argparse

import numpy as np

from leader_follower.boids_colony import BoidsColony
from leader_follower.bak.boid_spawner_bak import BoidSpawner
from leader_follower.bak.colony_helpers import StateBounds
from leader_follower.bak.file_helper import loadConfig
from leader_follower.bak.fitness_calculator_bak import FitnessCalculator
from leader_follower.bak.poi_colony import POIColony
from leader_follower.bak.poi_spawner_bak import POISpawner

from leader_follower.rewards import global_reward, calc_diff_rewards


def main(main_args):
    top_config = loadConfig(config_name='test_config.yaml')
    ccea_config = top_config['CCEA']
    env_run_config = ccea_config['config']['BoidsEnv']
    boids_config = env_run_config['config']

    # 50
    x_dim = boids_config["map_dimensions"]["x"]
    # 50
    y_dim = boids_config["map_dimensions"]["y"]
    # [50. 50.]
    map_dimensions = np.array([x_dim, y_dim], dtype=np.float64)
    # 'Density'
    poi_sensor_type = boids_config["ObservationManager"]['poi_sensor_type']
    # 'Density'
    swarm_sensor_type = boids_config["ObservationManager"]['swarm_sensor_type']
    ####################################################################################################################
    state_bounds = StateBounds(map_dimensions=map_dimensions, **boids_config["StateBounds"])

    boid_spawner = BoidSpawner(bounds=state_bounds, **boids_config["BoidSpawner"])
    poi_spawner = POISpawner(map_dimensions=map_dimensions, **boids_config["POISpawner"])

    boids_col = BoidsColony(init_state=boid_spawner.getSpawnState(), bounds=state_bounds, **boids_config["BoidsColony"])
    poi_col = POIColony(positions=poi_spawner.getSpawnPositions(), **boids_config["POIColony"])
    ####################################################################################################################
    fitness_calculator = FitnessCalculator(poi_col, boids_col)

    # 0.0
    team_fitness = fitness_calculator.getTeamFitness()
    # [0.0, 0.0, 0.0, 0.0]
    all_different_fitness = fitness_calculator.calculateDifferenceEvaluations()

    ref_fitness = global_reward(poi_col)
    ref_all_diff = calc_diff_rewards(boids_colony=boids_col, poi_colony=poi_col)

    assert team_fitness == ref_fitness
    assert np.array_equal(all_different_fitness, ref_all_diff)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
