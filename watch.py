import matplotlib.pyplot as plt
import pygame
from time import time, sleep
from sys import exit
import numpy as np

from lib.boids_env import BoidsEnv
from lib.network_lib import createNNfromWeights
from lib.ccea_lib import computeAction
from lib.file_helper import getLatestTrialName, loadTrial, loadConfig

PLOT_SCORES = True
PLAY_ENV = False
TRIALNAME = getLatestTrialName()
TRIALNAME = "trial_1048"

# Load in the trial data
save_data = loadTrial(TRIALNAME)
scores_list = save_data["scores_list"]
unfiltered_scores_list = save_data["unfiltered_scores_list"]
unfiltered_agent_scores_list = save_data["unfiltered_agent_scores_list"]
average_fitness_list_unfiltered = save_data["average_fitness_list_unfiltered"]
average_agent_fitness_lists_unfiltered = save_data["average_agent_fitness_lists_unfiltered"]
final_population = save_data["final_population"]
finished_iterations = save_data["finished_iterations"]
best_team_data = save_data["best_team_data"]

# Load in the config for that trial
config_filename = "config_" + TRIALNAME.split('_')[1] + ".yaml"
config = loadConfig(config_name=config_filename)
config["CCEA"]["config"]["BoidsEnv"]["render_mode"] = "human"
leader_colors = config["CCEA"]["config"]["BoidsEnv"]["config"]["Renderer"]["leader_colors"]
leader_colors = tuple(np.array(leader_colors)/255)

if PLOT_SCORES:
    plt.plot(scores_list, color="green", linestyle="--")
    plt.plot(unfiltered_scores_list, color="green")
    for ind, unfiltered_agent_scores in enumerate(unfiltered_agent_scores_list):
        plt.plot(unfiltered_agent_scores, color=tuple(leader_colors[ind%len(leader_colors)]))
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    # plt.ylim([0.0,1.2])
    plt.title("Best Performance")
    legend = ["Best Team Filtered", "Best Team Unfiltered"]
    for i in range(len(unfiltered_agent_scores_list)):
        legend.append("Agent "+str(i+1))
    plt.legend(legend)
    plt.show()



if PLOT_SCORES:
    plt.plot(average_fitness_list_unfiltered, color="green")
    for ind, unfiltered_agent_scores in enumerate(average_agent_fitness_lists_unfiltered):
        plt.plot(unfiltered_agent_scores, color=tuple(leader_colors[ind%len(leader_colors)]))
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    # plt.ylim([0.0,1.2])
    plt.title("Average Performance")
    legend = ["Average Team Performance"]
    for i in range(len(average_agent_fitness_lists_unfiltered)):
        legend.append("Agent "+str(i+1))
    plt.legend(legend)
    plt.show()


if PLAY_ENV:
    shutdown = False
    env = BoidsEnv(**config["CCEA"]["config"]["BoidsEnv"])
    dt = env.boids_colony.dt
    networks = [createNNfromWeights(genome_data.genome) for genome_data in best_team_data.team]
    print(best_team_data.fitness, best_team_data.all_evaluation_seeds)

    for eval_seed in best_team_data.all_evaluation_seeds:
        observations = env.reset(eval_seed)
        for step in range(env.max_steps):
            start_time = time()
            actions = {agent: computeAction(network, observations[agent], env) for agent, network in zip(env.possible_agents, networks)}
            observations, rewards, dones, infos = env.step(actions)
            env.render()
            # input()
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
        print("Team Fitness: ", env.fitness_calculator.getTeamFitness(), " | Agent Fitnesses: ", env.fitness_calculator.calculateDifferenceEvaluations())
