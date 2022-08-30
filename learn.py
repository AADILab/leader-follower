from lib.ccea_lib import CCEA
from time import time
import numpy as np
from lib.file_helper import loadPopulation, getNewTrialName, getLatestTrialName, saveTrial

NUM_GENERATIONS = 500
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

# Boid positions are randomized within the spawn radius of the spawn midpoint
# print("sp: ", self.spawn_radius)

num_leaders = 2
num_followers = 4

positions = np.array([
    # Follower positions
    [16,45],
    [14,45],
    [34.5,45],
    [36,45],
    # Leader positions
    [15,42.5],
    [35,42.5]
],dtype=float)

headings = -np.pi/2*np.ones((num_followers+num_leaders,1))

# spawn_radius = 5
# spawn_midpoint = np.array([25.,25.])

# rand_angles = np.random.uniform(low=0., high=2*np.pi, size=(num_followers+num_leaders,1))
# rand_radii = np.random.uniform(low=0.,high=spawn_radius,size=(num_followers+num_leaders,1))
# positions = np.hstack((
#     rand_radii*np.cos(rand_angles),
#     rand_radii*np.sin(rand_angles)
# )) + spawn_midpoint

# headings = np.random.uniform(low=0., high=2*np.pi, size = (num_followers+num_leaders, 1))

env_kwargs = {
    "num_leaders": num_leaders,
    "num_followers": num_followers,
    "FPS": 5,
    "num_steps": 10*5,
    "render_mode": 'none',
    "map_size": np.array([50,50]),
    "positions" : positions, #np.array([[25.,25.], [25, 26]]),
    "headings": headings,# np.array([[-np.pi/2], [np.pi/2]]),
    # "spawn_midpoint": np.array([25.,25.]),
    # "spawn_radius": 1,
    "spawn_velocity": 0,
    "poi_positions": np.array([[15.,30.], [35.,30.], [35.,35.], [15.,35.]]), # ,[20.,80.],[80.,20.],[80.,80.]
    "coupling": 3,
    "observe_followers": True}
learner = CCEA(num_agents = num_leaders, sub_population_size=15, num_parents=5, sigma_mutation=0.15, nn_hidden=[10], nn_outputs=2, num_workers=4, init_population=initial_population, use_difference_rewards=True, env_kwargs=env_kwargs)

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
