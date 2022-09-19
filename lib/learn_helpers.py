from lib.ccea_lib import CCEA
from time import time
import sys
from typing import Dict
import numpy as np
from lib.file_helper import saveTrial, loadConfig, setupInitialPopulation

def runExperiment(config: Dict) -> None:
    # Start clock
    start = time()

    # Setup learner
    learner = CCEA(**config["CCEA"], init_population=setupInitialPopulation(config))

    try:
        learner.train(num_generations=config["num_generations"])
    except KeyboardInterrupt:
        print("Program interrupted by user keyboard interrupt. Exiting program and saving experiment data.")

    learner.stop_event.set()

    best_fitness_list, final_population, finished_iterations, best_team_data = learner.getFinalMetrics()

    # Save data
    save_data = {
        "scores_list": best_fitness_list,
        "final_population": final_population,
        "finished_iterations": finished_iterations,
        "best_team_data": best_team_data
    }

    saveTrial(save_data, config)

    print("Experiment time: ", time() - start, " seconds. Completed ", finished_iterations, " out of ", config["num_generations"], " generations.")

