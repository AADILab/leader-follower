from time import time
import pygame

from env_lib import parallel_env, ROCK

def policy(observation, agent):
    return ROCK

FPS = 30
dt = 1/FPS

env = parallel_env(num_leaders = 0, num_followers = 20)
observations = env.reset()

last_time = None

# pygame.init()

shutdown = False
while not shutdown:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            shutdown = True
    current_time = time()
    if last_time is None or current_time - last_time >= dt:
        if last_time is not None: print("t:", current_time-last_time-dt)
        last_time = current_time
        env.step({})
        env.render()
