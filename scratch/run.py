from lib.env_lib import parallel_env, ROCK

def policy(observation, agent):
    return ROCK

env = parallel_env(num_leaders = 0, num_followers = 20)
observations = env.reset()
max_cycles = 100
for step in range(max_cycles):
    actions = {agent: policy(observations[agent], agent) for agent in env.agents}
    print(actions)
    observations, rewards, dones, infos = env.step(actions)
    env.render()



# def continuouslyRender(self, r, positions, headings):
#         delay_time = 0.1
#         last_time = -delay_time

#         shutdown = False
#         while not shutdown:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     shutdown = True
#                 current_time = time()
#                 if current_time - last_time > delay_time:
#                     last_time = current_time
#                     r.renderFrame(positions, headings)

# print(env.agents)

# print(env.headings)
# print(env.positions)