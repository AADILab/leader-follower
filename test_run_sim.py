from time import time
import pygame

from env_lib import parallel_env, ROCK

def policy(observation, agent):
    return ROCK

env = parallel_env(num_leaders = 0, num_followers = 50)
observations = env.reset()

delay_time = 1/60
last_time = None

# pygame.init()

shutdown = False
while not shutdown:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            shutdown = True
    current_time = time()
    if last_time is None or current_time - last_time >= delay_time:
        if last_time is not None: print(current_time-last_time-delay_time)
        last_time = current_time
        env.step({})
        env.render()
