import pickle
import matplotlib.pyplot as plt
from env_lib import BoidsEnv
from network_lib import createNNfromWeights
from learner_lib import computeAction
from time import time, sleep
from sys import exit
from file_helper import getLatestTrialName, loadTrial

import pygame


PLOT_SCORES = True
PLAY_ENV = True
TRIALNAME = getLatestTrialName()
RENDER_FOLLOWER_OBSERVATION = False
# TRIALNAME = "trial_68"

if not RENDER_FOLLOWER_OBSERVATION:
    f_inds = []
else:
    f_inds = None

save_data = loadTrial(TRIALNAME)
scores_list = save_data["scores_list"]
final_scores = save_data["final_scores"]
final_population = save_data["final_population"]
env_kwargs = save_data["env_kwargs"]
env_kwargs["follower_inds"] = f_inds

if PLOT_SCORES:
    plt.plot(scores_list)
    plt.show()


if PLAY_ENV:
    shutdown = False
    env_kwargs["render_mode"] = "human"
    env = BoidsEnv(**env_kwargs)
    env.renderer.render_POIs = True
    refresh_time = 1/env.FPS
    for count, genome in enumerate(final_population[:]):
        print("Playing genome ",count+1)
        network = createNNfromWeights(genome)
        print("Number of weights in network: ", network.total_weights)
        observations = env.reset()
        for _ in range(env.num_steps):
            start_time = time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Shutdown command recieved. Shutting down.")
                    shutdown = True
            if shutdown: exit()
            actions = {agent: computeAction(network, observations[agent], env) for agent in env.possible_agents}
            observations, rewards, dones, infos = env.step(actions)
            env.render()
            loop_time = time() - start_time
            if loop_time < refresh_time:
                sleep(refresh_time - loop_time)
            else:
                print("Loop took longer than refresh rate")
