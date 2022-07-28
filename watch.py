import pickle
import matplotlib.pyplot as plt
from env_lib import BoidsEnv
from network_lib import createNNfromWeights
from learner_lib import computeAction
from time import time, sleep
from sys import exit

import pygame


PLOT_SCORES = True
PLAY_ENV = True
FILENAME = "trial_3.pkl"

save_data = pickle.load(open(FILENAME, "rb"))
scores_list = save_data["scores_list"]
final_scores = save_data["final_scores"]
final_population = save_data["final_population"]
env_kwargs = save_data["env_kwargs"]

# print("scores_list: ", scores_list)
if PLOT_SCORES:
    plt.plot(scores_list)
    plt.show()


if PLAY_ENV:
    shutdown = False
    env_kwargs["render_mode"] = "human"
    env = BoidsEnv(**env_kwargs)
    refresh_time = 1/env.FPS
    for count, genome in enumerate(final_population[:]):
        print("Playing genome ",count)
        network = createNNfromWeights(genome)
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
