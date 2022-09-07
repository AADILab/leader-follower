import matplotlib.pyplot as plt
from lib.env_lib import BoidsEnv
from lib.network_lib import createNNfromWeights
from lib.learner_lib import computeAction
from time import time, sleep
from sys import exit
from lib.file_helper import getLatestTrialName, loadTrial

import pygame


PLOT_SCORES = True
PLAY_ENV = True
TRIALNAME = getLatestTrialName()
RENDER_FOLLOWER_OBSERVATION = False
RENDER_LEADER_OBSERVATION = False
# TRIALNAME = "trial_534"
# TRIALNAME = "trial_149"
# TRIALNAME = "trial_412"

if not RENDER_FOLLOWER_OBSERVATION:
    f_inds = []
else:
    f_inds = None

save_data = loadTrial(TRIALNAME)
scores_list = save_data["scores_list"]
final_population = save_data["final_population"]
best_team_data = save_data["best_team_data"]
env_kwargs = save_data["env_kwargs"]
env_kwargs["follower_inds"] = f_inds
# env_kwargs["observe_followers"] = True

if PLOT_SCORES:
    plt.plot(scores_list)
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.ylim([0.0,1.2])
    plt.title("Team Performance")
    plt.show()


if PLAY_ENV:
    shutdown = False
    env_kwargs["render_mode"] = "human"
    env = BoidsEnv(**env_kwargs)
    # env.renderer.render_POIs = True
    # env.renderer.render_centroid_observations = False
    # env.renderer.render_POI_observations = False
    refresh_time = 1/env.FPS
    networks = [createNNfromWeights(genome_data.genome) for genome_data in best_team_data.team]

    # for count, genome in enumerate(final_population[:]):
    # print("Playing genome ",count+1," with score ", final_scores[count])
    # network = createNNfromWeights(genome)
    # print("Number of weights in network: ", network.total_weights)
    observations = env.reset()
    for _ in range(env.num_steps):
        start_time = time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Shutdown command recieved. Shutting down.")
                shutdown = True
        if shutdown: exit()
        actions = {agent: computeAction(network, observations[agent], env) for agent, network in zip(env.possible_agents, networks)}
        observations, rewards, dones, infos = env.step(actions)
        env.render(kwargs={"render_centroid_observations": RENDER_LEADER_OBSERVATION})
        loop_time = time() - start_time
        if loop_time < refresh_time:
            sleep(refresh_time - loop_time)
        else:
            print("Loop took longer than refresh rate")
    # print("Final Score: ", rewards["team"])
