from lib.ccea_lib import CCEA
from time import time
import numpy as np
from lib.file_helper import loadPopulation, getNewTrialName, getLatestTrialName, saveTrial

NUM_GENERATIONS = 100
EXPERIMENT_NAME = getNewTrialName()
LOAD_POPULATION = None
# LOAD_POPULATION = getLatestTrialName()
# LOAD_POPULATION = "trial_302"

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
env_kwargs = {
    "num_leaders": 2,
    "num_followers": 0,
    "FPS": 5,
    "num_steps": 10*5,
    "render_mode": 'none',
    "map_size": np.array([50,50]),
    "positions" : np.array([[25.,25.], [25, 26]]),
    "headings": np.array([[-np.pi/2], [np.pi/2]]),
    # "spawn_midpoint": np.array([25.,25.]),
    # "spawn_radius": 1,
    "spawn_velocity": 0,
    "poi_positions": np.array([[15.,15.], [35.,15.], [35.,35.], [15.,35.]]), # ,[20.,80.],[80.,20.],[80.,80.]
    "coupling": 2,
    "observe_followers": False}
learner = CCEA(num_agents = 2, sub_population_size=15, num_parents=5, sigma_mutation=0.15, nn_hidden=[10], nn_outputs=2, num_workers=4, init_population=initial_population, env_kwargs=env_kwargs)

try:
    learner.train(num_generations=NUM_GENERATIONS)
except KeyboardInterrupt:
    print("Program interrupted by user keyboard interrupt. Exiting program and saving experiment data.")

learner.stop_event.set()

best_fitness_list, final_population, finished_iterations, best_team_data = learner.getFinalMetrics()

# Save data
save_data = {
    "scores_list": best_fitness_list,
    "final_population": final_population,
    "finished_iterations": finished_iterations,
    "best_team_data": best_team_data,
    "env_kwargs": env_kwargs
}

saveTrial(save_data)

print("Experiment time: ", time() - start, " seconds. Completed ", finished_iterations, " out of ", NUM_GENERATIONS, " generations.")
