from learner_lib import Learner
from time import time
import numpy as np
import pickle

start_positions = np.array([[40.,40.]])
start_velocities = np.array([[0.]])
start_headings = np.array([[-np.pi/2]])

start = time()
env_kwargs = {"num_leaders": 1, "num_followers": 0, "FPS": 60, "num_steps": 10*60, "render_mode": 'none'}
learner = Learner(population_size=15, num_parents=5, sigma_mutation=0.25, env_kwargs={"num_leaders": 1, "num_followers": 0, "FPS": 60, "num_steps": 10*60, "render_mode": 'none', "positions": start_positions, "velocities": start_velocities, "headings": start_headings})

try:
    learner.train(num_generations=1000)
except KeyboardInterrupt:
    print("Program interrupted by user keyboard interrupt. Exiting program and saving experiment data.")

scores_list, final_scores, final_population = learner.getFinalMetrics()

learner.cleanup()

# Save data
pickle.dump((scores_list, final_scores, final_population, env_kwargs), open("trial_0.pkl", "wb"))

print("Experiment time: ", time() - start, " seconds")
