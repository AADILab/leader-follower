from lib.ccea_lib import CCEA
from time import time
import numpy as np
from lib.file_helper import loadPopulation, getNewTrialName, getLatestTrialName, saveTrial
import myaml

# Load in config
config = myaml.safe_load("configs/default.yaml")

# NUM_GENERATIONS = 500
# EXPERIMENT_NAME = getNewTrialName()
# LOAD_POPULATION = None
# LOAD_POPULATION = getLatestTrialName()
# LOAD_POPULATION = "trial_302"

if config["load_population"] is not None:
    if config["load_population"] == "latest":
        config["load_population"] = getLatestTrialName()
    initial_population = loadPopulation(config["load_population"])
else:
    initial_population = None

if config["experiment_name"] is None:
    config["experiment_name"] = getNewTrialName()

filename = config["experiment_name"] + ".pkl"

start = time()

learner = CCEA(**config["CCEA"])
#     num_agents = num_leaders,
#     sub_population_size=15,
#     num_parents=5,
#     mutation_rate=0.15,
#     mutation_probability=0.15,
#     nn_hidden=[10],
#     nn_outputs=2,
#     num_workers=4,
#     init_population=initial_population,
#     use_difference_rewards=True,
#     env_kwargs=env_kwargs
# )

# try:
#     learner.train(num_generations=NUM_GENERATIONS)
# except KeyboardInterrupt:
#     print("Program interrupted by user keyboard interrupt. Exiting program and saving experiment data.")

# learner.stop_event.set()

# best_fitness_list, final_population, finished_iterations, best_team_data = learner.getFinalMetrics()

# # Save data
# save_data = {
#     "scores_list": best_fitness_list,
#     "final_population": final_population,
#     "finished_iterations": finished_iterations,
#     "best_team_data": best_team_data,
#     "env_kwargs": env_kwargs
# }

# saveTrial(save_data)

# print("Experiment time: ", time() - start, " seconds. Completed ", finished_iterations, " out of ", NUM_GENERATIONS, " generations.")
