import sys

import pygame; sys.path.append("/home/egonzalez/leaders")
from lib.boids_env import env, parallel_env
import myaml
import numpy as np

from time import sleep

config=myaml.safe_load("configs/default.yaml")

ParallelEnv = True
if ParallelEnv:
    # Instantiate parallel env directly
    my_env = parallel_env(**config["env"])
    my_env.reset()
    # print()

    while my_env.renderer is None or not my_env.renderer.checkForPygameQuit():
        observations, rewards, dones, infos = my_env.step({agent: 100*np.ones(2) for agent in my_env.agents})
        my_env.render(mode="human")
        sleep(1/60)

# Agent Environment Cycle
else:
    my_env = env(config["env"])
    my_env.reset()
    shutdown = False
    # pygame.init()
    while not shutdown:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         shutdown = True
        for agent in my_env.agent_iter():
            observation, reward, done, info = my_env.last()
            action = 100*np.ones(2)
            my_env.step(action)
            my_env.render(mode="human")
        sleep(1/60)
