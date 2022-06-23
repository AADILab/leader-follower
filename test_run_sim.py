from time import time
import pygame

from env_lib import BoidsEnv, ROCK

import numpy as np
np.random.seed(3)

def policy(observation, agent):
    return ROCK

positions = np.vstack((
    np.hstack((
                np.random.uniform(50, size=(20,1)),
                np.random.uniform(50, size=(20,1))
            )),
    np.ones((10,2))*25
))
positions = None

env = BoidsEnv(num_leaders = 1, num_followers = 50, FPS=30, positions=positions, r_ind=[0])
observations = env.reset()

dt = env.bm.dt
last_time = None

shutdown = False
while not shutdown:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            shutdown = True
    current_time = time()
    if last_time is None or current_time - last_time >= dt:
        # if last_time is not None: print("t:", current_time-last_time-dt)
        last_time = current_time
        observations, rewards, dones, infos = env.step({})
        # print("leader 1:", observations["leader_1"])
        # print("leader 2:", observations["leader_2"])
        # print("leader 3:", observations["leader_3"])
        # print("leader 4:", observations["leader_4"])
        # print("leader 5:", observations["leader_5"])
        env.render()
