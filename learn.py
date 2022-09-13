from lib.ccea_lib import CCEA
from time import time
import numpy as np
from lib.file_helper import loadPopulation, getLatestTrialName, saveTrial
import myaml

# Load in config
config = myaml.safe_load("configs/default.yaml")

if config["load_population"] is not None:
    if config["load_population"] == "latest":
        config["load_population"] = getLatestTrialName()
    initial_population = loadPopulation(config["load_population"])
else:
    initial_population = None

start = time()

learner = CCEA(**config["CCEA"])

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
