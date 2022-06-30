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

env = BoidsEnv(num_leaders = 3, num_followers = 50, FPS=30, positions=positions, follower_inds=[])
observations = env.reset()

dt = env.bm.dt
last_time = None

count = 0
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
        env.render()
    count += 1
    # if count > 0:
    #     shutdown = True