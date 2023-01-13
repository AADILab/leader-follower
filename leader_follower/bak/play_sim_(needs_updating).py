# todo
# from time import time
#
# import numpy as np
# import pygame
# from leader_follower.boids_env import BoidsEnv
#
# np.random.seed(3)
#
#
# def policy(observation, agent):
#     pressed_keys = pygame.key.get_pressed()
#     if pressed_keys[pygame.K_UP]:
#         vel = 10
#     elif pressed_keys[pygame.K_DOWN]:
#         vel = 0
#     else:
#         vel = 5
#     if pressed_keys[pygame.K_RIGHT]:
#         turn = -np.pi
#     elif pressed_keys[pygame.K_LEFT]:
#         turn = np.pi
#     else:
#         turn = 0
#     return np.array([turn, vel])
#
#
# num_followers = 100
# num_leaders = 10
# total_agents = num_followers + num_leaders
# map_size = np.array([100, 100])
#
# # Setup positions so leaders are all centered
# positions = np.vstack((
#     np.hstack((
#         np.random.uniform(map_size[0], size=(num_followers, 1)),
#         np.random.uniform(map_size[1], size=(num_followers, 1))
#     )),
#     np.hstack((
#         np.ones((num_leaders, 1)) * map_size[0] / 2,
#         np.ones((num_leaders, 1)) * map_size[1] / 2
#     ))
# ))
# # Modify leader positions so they are clustered around the same location
# positions[num_followers:, 0] += np.random.uniform(num_leaders, size=num_leaders) - num_leaders / 2
# positions[num_followers:, 1] += np.random.uniform(num_leaders, size=num_leaders) - num_leaders / 2
#
# # Setup headings and modify them so all leaders are aligned
# headings = np.random.uniform(0, 2 * np.pi, size=(num_followers + num_leaders, 1))
# headings[num_followers:] = np.pi / 2
#
# env_kwargs = {"num_leaders": 1, "num_followers": 3, "FPS": 60, "num_steps": 60 * 60, "render_mode": "human",
#               "map_size": np.array([100, 100]), "spawn_midpoint": np.array([50., 50.]), "spawn_radius": 2,
#               "spawn_velocity": 0, "poi_positions": np.array(
#         [[20., 20.], [20., 80.], [80., 20.], [80., 80.]])}  # ,[20.,80.],[80.,20.],[80.,80.]
#
# # env = BoidsEnv(num_leaders = num_leaders, num_followers = num_followers, FPS=60, num_steps=60*60,
# # follower_inds=[], render_mode='human', map_size=map_size, positions = None, headings = headings)
# env = BoidsEnv(**env_kwargs)
#
# # Reset env for first round of observations
# observations = env.reset()
#
# last_time = None
# shutdown = False
# while not shutdown:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             shutdown = True
#     current_time = time()
#     if last_time is None or current_time - last_time >= env.dt:
#         # if last_time is not None: print("t:", current_time-last_time-dt)
#         last_time = current_time
#         actions = {agent: policy(observations[agent], agent) for agent in env.possible_agents}
#         observations, rewards, dones, infos = env.step(actions)
#         env.render()
#         # print(rewards["team"])
#         print([str(n).split(".")[0] for n in observations["leader_1"]])
