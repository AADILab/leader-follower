import pickle
import matplotlib.pyplot as plt
from env_lib import BoidsEnv
from network_lib import createNNfromWeights
from time import time, sleep
from sys import exit

import pygame


PLOT_SCORES = False
PLAY_ENV = True

save_data = pickle.load(open("trial_1.pkl", "rb"))
scores_list = save_data["scores_list"]
final_scores = save_data["final_scores"]
final_population = save_data["final_population"]
env_kwargs = save_data["env_kwargs"]

print("scores_list: ", scores_list)
if PLOT_SCORES:
    plt.plot(scores_list)
    plt.show()


if PLAY_ENV:
    shutdown = False
    env_kwargs["render_mode"] = "human"
    env = BoidsEnv(**env_kwargs)
    refresh_time = 1/env.FPS
    for count, genome in enumerate(final_population[:1]):
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
            actions = {agent: network.forward(observations[agent]) for agent in env.possible_agents}
            observations, rewards, dones, infos = env.step(actions)
            env.render()
            loop_time = time() - start_time
            if loop_time < refresh_time:
                sleep(refresh_time - loop_time)
            else:
                print("Loop took longer than refresh rate")
