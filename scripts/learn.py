from pathlib import Path
from time import time
from typing import Dict

from leader_follower import project_properties
from leader_follower.bak.file_helper import load_config
from leader_follower.bak.file_helper import save_trial, setup_initial_population
from leader_follower.bak.ccea_lib import CCEA


def run_experiment(base_dir, config: Dict) -> None:
    # Start clock
    start = time()

    # Setup learner
    initial_population = setup_initial_population(base_dir, config)
    learner = CCEA(**config["CCEA"], init_population=initial_population)

    try:
        learner.train(num_generations=config["num_generations"])
    except KeyboardInterrupt:
        print("Program interrupted by user keyboard interrupt. Exiting program and saving experiment data.")

    best_fitness_list, best_fitness_list_unfiltered, best_agent_fitness_lists_unfiltered, \
        average_fitness_list_unfiltered, average_agent_fitness_lists_unfiltered, \
        final_population, finished_iterations, best_team_data = learner.final_metrics()

    # Save data
    save_data = {
        "scores_list": best_fitness_list,
        "unfiltered_scores_list": best_fitness_list_unfiltered,
        "unfiltered_agent_scores_list": best_agent_fitness_lists_unfiltered,
        "average_fitness_list_unfiltered": average_fitness_list_unfiltered,
        "average_agent_fitness_lists_unfiltered": average_agent_fitness_lists_unfiltered,
        "final_population": final_population,
        "finished_iterations": finished_iterations,
        "best_team_data": best_team_data
    }

    save_trial(base_dir, save_data, config)

    print("Experiment time: ", time() - start, " seconds. Completed ", finished_iterations, " out of ",
          config["num_generations"], " generations.")
    return

def main():
    """
spawn_rule: Individual
position_rule: Set
velocity_rule: FixedStart
heading_rule: FixedStart
velocity_fraction: 0.0
start_heading: !eval 3.14159/2
dt: !eval 1/5
radius_attraction: !eval 5
radius_orientation: !eval 1.000000001
radius_repulsion: !eval   1
attraction_multiplier: 5
orientation_multiplier: 0
repulsion_multiplier: 1
wall_avoidance_multiplier: 1
num_poi_bins: 4
num_swarm_bins: 4
observation_rule: Individual
poi_sensor_type: Density
swarm_sensor_type: Density
observation_radius: ~
full_observability: true
poi_coupling: 3
poi_observation_radius: 5
poi_spawn_rule: Set
pixels_per_unit: 20
max_acceleration: 5
max_angular_velocity: 1.570795
max_velocity: 10
min_velocity: 0
map_dimensions:
  x: 50
  y: 50
reward_type: difference_leader
init_seed: 1
max_steps: !eval 5*5
render_mode: none
mutation_probability: 0.15
mutation_rate: 0.1
nn_hidden:
  - 10
num_evaluations: 5
num_workers: 8
sub_population_size: 25
use_difference_rewards: true
load_population: null
num_generations: 20
    :return:
    """
    config_fns = [each_fn for each_fn in Path(project_properties.config_dir).rglob('*.yaml')]

    config_name = 'default.yaml'
    subpop_size = 50
    n_gens = 100
    stat_runs = 5
    for each_experiment in config_fns:
        print(f'{"=" * 80}')
        print(f'{each_experiment}')
        config = load_config(each_experiment)
        config['CCEA']['sub_population_size'] = subpop_size
        config['num_generations'] = n_gens
        # todo get stat_runs from config
        # Run each experiment n times
        for idx in range(stat_runs):
            print(f'Running experiment {idx}')
            run_experiment(each_experiment, config)
        print(f'{"=" * 80}')
    return


if __name__ == '__main__':
    main()
