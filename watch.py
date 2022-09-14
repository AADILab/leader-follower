import matplotlib.pyplot as plt
import pygame
from time import time, sleep
from sys import exit

from lib.boids_env import BoidsEnv
from lib.network_lib import createNNfromWeights
from lib.ccea_lib import computeAction
from lib.file_helper import getLatestTrialName, loadTrial, loadConfig

PLOT_SCORES = False
PLAY_ENV = True
TRIALNAME = getLatestTrialName()

# Load in the trial data
save_data = loadTrial(TRIALNAME)
scores_list = save_data["scores_list"]
final_population = save_data["final_population"]
finished_iterations = save_data["finished_iterations"]
best_team_data = save_data["best_team_data"]

# Load in the config for that trial
config_filename = "config_" + TRIALNAME.split('_')[1] + ".yaml"
config = loadConfig(config_name=config_filename)
config["CCEA"]["config"]["BoidsEnv"]["render_mode"] = "human"

if PLOT_SCORES:
    plt.plot(scores_list)
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.ylim([0.0,1.2])
    plt.title("Team Performance")
    plt.show()


if PLAY_ENV:
    shutdown = False
    env = BoidsEnv(**config["CCEA"]["config"]["BoidsEnv"])
    dt = env.boids_colony.dt
    networks = [createNNfromWeights(genome_data.genome) for genome_data in best_team_data.team]
    print(best_team_data.fitness, best_team_data.evaluation_seed)

    observations = env.reset(best_team_data.evaluation_seed)
    for step in range(env.max_steps):
        start_time = time()
        actions = {agent: computeAction(network, observations[agent], env) for agent, network in zip(env.possible_agents, networks)}
        observations, rewards, dones, infos = env.step(actions)
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Shutdown command recieved. Shutting down.")
                shutdown = True
        if shutdown: exit()
        loop_time = time() - start_time
        if loop_time < dt:
            sleep(dt - loop_time)
        else:
            print("Loop " + str(step) + " took longer than refresh rate")
    print(env.fitness_calculator.getTeamFitness())
