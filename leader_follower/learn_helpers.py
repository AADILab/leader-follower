from time import time
from typing import Dict

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

    learner.stop_event.set()

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
