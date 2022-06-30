from time import time
import pygame

from env_lib import BoidsEnv, ROCK

import numpy as np
np.random.seed(3)

def policy(observation, agent):
    return np.array([-np.pi,0])

positions = np.vstack((
    np.hstack((
                np.random.uniform(50, size=(20,1)),
                np.random.uniform(50, size=(20,1))
            )),
    np.ones((10,2))*25
))
positions = None

env = BoidsEnv(num_leaders = 1, num_followers = 100, FPS=50, positions=positions, follower_inds=[7,8])
# Set leader velocities to zero
env.bm.velocities[env.bm.num_followers:] = 0
observations = env.reset()

last_time = None
count = 0
shutdown = False
while not shutdown:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            shutdown = True
    current_time = time()
    if last_time is None or current_time - last_time >= env.dt:
        # if last_time is not None: print("t:", current_time-last_time-dt)
        last_time = current_time
        actions = {agent: policy(observations[agent], agent) for agent in env.possible_agents}
        # print("a: ", actions)
        observations, rewards, dones, infos = env.step(actions)
        env.render()
    count += 1
