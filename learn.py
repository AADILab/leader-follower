from learner_lib import Learner
from time import time
import numpy as np
import pickle
from file_helper import loadPopulation, getNewTrialName

NUM_GENERATIONS = 30
EXPERIMENT_NAME = getNewTrialName()
LOAD_POPULATION = None

if LOAD_POPULATION is not None:
    initial_population = loadPopulation(LOAD_POPULATION)
else:
    initial_population = None

filename = EXPERIMENT_NAME + ".pkl"

start_positions = np.array([[40.,40.], [10,40],[40,10]])
start_velocities = np.array([[0.,0.,0.]]).T
start_headings = np.array([[-np.pi/2,-np.pi/2,-np.pi]]).T

start = time()
env_kwargs = {"num_leaders": 3, "num_followers": 0, "FPS": 5, "num_steps": 6*5, "render_mode": 'none', "positions": start_positions, "velocities": start_velocities, "headings": start_headings}
learner = Learner(population_size=100, num_parents=5, sigma_mutation=0.1, nn_inputs=2, nn_hidden=5, nn_outputs=2, init_population = initial_population, env_kwargs=env_kwargs)

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
pickle.dump(save_data, open(filename, "wb"))

print("Experiment time: ", time() - start, " seconds. Completed ", finished_iterations, " out of ", NUM_GENERATIONS, " generations.")
