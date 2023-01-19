from pathlib import Path
from time import time

from leader_follower import project_properties
from leader_follower.bak.ccea_lib import CCEA
from leader_follower.bak.file_helper import load_config
from leader_follower.bak.file_helper import save_trial, setup_initial_population


def run_experiment(base_dir, positions, meta_params):
    # Start clock
    start = time()

    # Setup learner
    initial_population = setup_initial_population(base_dir, meta_params)
    learner = CCEA(**positions, **meta_params, init_population=initial_population)

    try:
        learner.train(num_generations=meta_params["num_generations"])
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
    config_fns = [each_fn for each_fn in Path(project_properties.config_dir).rglob('*.yaml')]
    subpop_size = 50
    n_gens = 100
    stat_runs = 5

    config_name = Path(project_properties.config_dir, 'meta_params.yaml')
    meta_params = load_config(config_name)

    meta_params['sub_population_size'] = subpop_size
    meta_params['num_generations'] = n_gens
    for each_experiment in config_fns:
        print(f'{"=" * 80}')
        print(f'{each_experiment}')
        positions = load_config(each_experiment)

        # todo get stat_runs from config
        # Run each experiment n times
        for idx in range(stat_runs):
            print(f'Running experiment {idx}')
            run_experiment(each_experiment, positions, meta_params)
        print(f'{"=" * 80}')
    return


if __name__ == '__main__':
    main()
