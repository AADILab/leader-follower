from env_lib import parallel_env, ROCK

def policy(observation, agent):
    return ROCK

env = parallel_env()
observations = env.reset()
max_cycles = 1
for step in range(max_cycles):
    actions = {agent: policy(observations[agent], agent) for agent in env.agents}
    # print(actions)
    observations, rewards, dones, infos = env.step(actions)

# print(env.agents)

# print(env.headings)
# print(env.positions)