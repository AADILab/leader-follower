from pathlib import Path
from time import time
from typing import Dict

from leader_follower import project_properties
from leader_follower.file_helper import load_config
from leader_follower.project_properties import data_dir
from leader_follower.learn.ccea_lib import CCEA
from leader_follower.file_helper import save_trial, setup_initial_population


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

    # learner.stop_event.set()

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
    config_fns = [each_fn for each_fn in Path(project_properties.config_dir).rglob('*.yaml')]

    config_name = 'default.yaml'
    subpop_size = 50
    n_gens = 100
    stat_runs = 5
    for each_experiment in config_fns:
        # try:
        print(f'{"=" * 80}')
        print(f'{each_experiment}')
        config = load_config(each_experiment.parent, config_name=config_name)
        config['CCEA']['sub_population_size'] = subpop_size
        config['num_generations'] = n_gens
        # todo get stat_runs from config
        # Run each experiment n times
        for idx in range(stat_runs):
            print(f'Running experiment {idx}')
            run_experiment(each_experiment, config)
        print(f'{"=" * 80}')
        # except Exception as e:
        #     print(f'Experiment {each_experiment}\n'
        #           f'{e}')
        #     print(f'{"=" * 80}')
    return


if __name__ == '__main__':
    main()
