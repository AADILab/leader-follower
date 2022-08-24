from lib.learner_lib import Learner
from time import time
import numpy as np
from lib.file_helper import loadPopulation, getNewTrialName, getLatestTrialName, saveTrial

NUM_GENERATIONS = 100
EXPERIMENT_NAME = getNewTrialName()
LOAD_POPULATION = None
# LOAD_POPULATION = getLatestTrialName()
# LOAD_POPULATION = "trial_167"

if LOAD_POPULATION is not None:
    initial_population = loadPopulation(LOAD_POPULATION)
else:
    initial_population = None

filename = EXPERIMENT_NAME + ".pkl"

# start_positions = np.hstack((
#                 22.5+np.random.uniform(5, size=(4,1)),
#                 22.5+np.random.uniform(5, size=(4,1))
#             ))
# start_velocities = np.zeros((4,1))
# start_headings = np.random.uniform(0, 2*np.pi, size=(4,1))

start = time()
# env_kwargs = {"num_leaders": 1, "num_followers": 3, "FPS": 5, "num_steps": 10*5, "render_mode": 'none', "positions": start_positions, "velocities": start_velocities, "headings": start_headings}
env_kwargs = {"num_leaders": 1, "num_followers": 0, "FPS": 5, "num_steps": 10*5, "render_mode": 'none', "map_size": np.array([100,100]),
    "spawn_midpoint": np.array([50,50]), "spawn_radius": 0.1, "spawn_velocity": 0, "headings": np.array([[-np.pi]]),
    "poi_positions": np.array([[20,20]])}
learner = Learner(population_size=15, num_parents=5, sigma_mutation=0.15, nn_inputs=env_kwargs["poi_positions"].size+2, nn_hidden=[10], nn_outputs=2, init_population = initial_population, env_kwargs=env_kwargs)

try:
    learner.train(num_generations=NUM_GENERATIONS)
except KeyboardInterrupt:
    print("Program interrupted by user keyboard interrupt. Exiting program and saving experiment data.")

learner.stop_event.set()

scores_list, final_scores, final_population, finished_iterations = learner.getFinalMetrics()

# Save data
save_data = {
    "scores_list": scores_list,
    "final_scores": final_scores,
    "final_population": final_population,
    "finished_iterations": finished_iterations,
    "env_kwargs": env_kwargs
}

saveTrial(save_data)

print("Experiment time: ", time() - start, " seconds. Completed ", finished_iterations, " out of ", NUM_GENERATIONS, " generations.")
