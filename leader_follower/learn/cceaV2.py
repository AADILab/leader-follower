"""
@title

@description

"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import trange

from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork


def select_roulette(population, select_size=1):
    fitness_vals = [pop['fitness'] for pop in population]

    # add small amount of noise to each fitness value (help deal with all same value)
    noise = np.random.uniform(0, 1, len(fitness_vals))
    fitness_vals += noise

    # if there is one positive fitness, then the others all being negative causes issues when calculating the probs
    fitness_norms = (fitness_vals - np.min(fitness_vals)) / (np.max(fitness_vals) - np.min(fitness_vals))
    if np.isnan(fitness_norms).any():
        fitness_norms = fitness_vals
    fitness_probs = fitness_norms / np.sum(fitness_norms)

    sim_pop = np.random.choice(population, select_size, replace=False, p=fitness_probs)
    return sim_pop


def mutate_gaussian(individual, proportion=1, amount=0.05):
    model = individual['network']
    model_copy = copy.deepcopy(model)

    # todo  add small amount of noise to each value
    #       add more noise to fewer values
    with torch.no_grad():
        param_vector = parameters_to_vector(model_copy.parameters())

        n_params = len(param_vector)
        noise = Normal(0, 1).sample(torch.Size((n_params,)))
        param_vector.add_(noise)

        vector_to_parameters(param_vector, model_copy.parameters())
    new_ind = {
        'network': model_copy,
        'fitness': None
    }
    return new_ind


# def get_action(self, state):
#     with torch.no_grad():
#         pred = self(state)
#
#     pred_probab = nn.Softmax(dim=0)(pred)
#     y_pred = pred_probab.argmax()
#     # convert from tensor to action
#     action_val = y_pred.item()
#     action = list(Action.__members__)[action_val]
#     action = Action[action]
#     return action

def get_action(net, observation, env):
    # todo fix to use pytorch backend
    out = net.forward(observation)
    # Map [-1,+1] to [-pi,+pi]
    heading = out[0] * np.pi
    # Map [-1,+1] to [0, max_velocity]
    velocity = (out[1] + 1.0) / 2 * env.state_bounds.max_velocity
    return np.array([heading, velocity])

def rollout(env: LeaderFollowerEnv, individuals, reward_func, render=False):
    env.reset()
    agent_dones = env.done()
    done = all(agent_dones.values())

    # select policy to use for each learning agent
    for agent_name, policy_info in individuals.items():
        env.agent_mapping[agent_name].policy = policy_info['network']

    while not done:
        observations = env.get_observations()
        next_actions = env.get_actions_from_observations(observations=observations)
        observations, rewards, agent_dones, truncs, infos = env.step(next_actions)
        done = all(agent_dones.values())
        if render:
            env.render()

    episode_rewards = reward_func(env)

    if type(episode_rewards) == float:
        # print(agent_rewards, len(env._leaders)+env._followers))
        # agent_rewards = [agent_rewards] * (len(env._leaders)+len(env._followers))
        episode_rewards = {
            agent_name: episode_rewards for agent_name, _ in individuals.items()
        }

    return episode_rewards

    # episode_reward = [global_reward for _ in env.agents]
    # episode_reward = env.agent_episode_rewards()
    # if reward_type == 'global':
    #     episode_reward = [global_reward for _ in episode_reward]
    # return episode_reward


def downselect_top_n(population, max_size):
    sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
    return sorted_pop[:max_size]


def neuro_evolve(env, n_hidden, population_size, n_gens, sim_pop_size, reward_func):
    debug = False
    select_func = select_roulette
    mutate_func = mutate_gaussian
    downselect_func = downselect_top_n

    # only creat sub-pops for agents capable of learning
    agent_pops = {
        agent_name: [
            {
                'network': NeuralNetwork(
                    n_inputs=env.agent_mapping[agent_name].n_in,
                    n_hidden=n_hidden,
                    n_outputs=env.agent_mapping[agent_name].n_out,
                ),
                'fitness': None
            }
            for _ in range(population_size)
        ]
        for agent_name in env.agents
        if env.agent_mapping[agent_name].type == 'learner'
    }
    print(f'Using device: {list(agent_pops.values())[0][0]["network"].device()}')

    # initial fitness evaluation of all networks in population
    # todo check reward structure/assignment to make sure reward is being properly assigned
    for pop_idx in range(population_size):
        new_inds = {agent_name: policy_info[pop_idx] for agent_name, policy_info in agent_pops.items()}
        agent_rewards = rollout(env, new_inds, reward_func=reward_func, render=debug)
        for agent_name, policy_info in agent_pops.items():
            policy_fitness = agent_rewards[agent_name]
            policy_info[pop_idx]['fitness'] = policy_fitness

    max_fitnesses = []
    avg_fitnesses = []
    for gen_idx in trange(n_gens):
        fitnesses = [
            [
                policy_info[pop_idx]['fitness']
                for agent_name, policy_info in agent_pops.items()
            ]
            for pop_idx in range(population_size)
        ]
        max_fitnesses.append(np.max(fitnesses, axis=0))
        avg_fitnesses.append(np.average(fitnesses, axis=0))
        sim_pops = [
            select_func(policy_population, sim_pop_size)
            for agent_name, policy_population in agent_pops.items()
        ]
        # todo multiprocess simulating each simulation population
        for sim_pop_idx, each_ind in enumerate(sim_pops):
            new_inds = {
                agent_name: mutate_func(policy_info[sim_pop_idx])
                for agent_name, policy_info in agent_pops.items()
            }

            # rollout and evaluate
            agent_rewards = rollout(env, new_inds, reward_func=reward_func, render=debug)
            for agent_name, policy_info in new_inds.items():
                policy_fitness = agent_rewards[agent_name]
                policy_info['fitness'] = policy_fitness
                # reinsert new individual into population of policies
                agent_pops[agent_name].append(policy_info)

        # downselect
        agent_pops = {
            agent_name: downselect_func(policy_info, population_size)
            for agent_name, policy_info in agent_pops.items()
        }

    best_policies = {}
    for agent_name, policy_info in agent_pops.items():
        fitness_vals = [pop['fitness'] for pop in policy_info]
        arg_best = np.argmax(fitness_vals)
        best_ind = policy_info[arg_best]
        print(best_ind['fitness'])
        best_policies[agent_name] = best_ind
    return best_policies, max_fitnesses, avg_fitnesses


def plot_fitnesses(avg_fitnesses, max_fitnesses, xtag=None, ytag=None):
    fig, axes = plt.subplots()

    avg_fitnesses = np.transpose(avg_fitnesses)
    max_fitnesses = np.transpose(max_fitnesses)

    for idx, each_fitness in enumerate(avg_fitnesses):
        axes.plot(each_fitness, label=f'Avg: Agent {idx+1}')

    for idx, each_fitness in enumerate(max_fitnesses):
        axes.plot(each_fitness, label=f'Max: Agent {idx+1}')

    axes.xaxis.grid()
    axes.yaxis.grid()
    axes.legend(loc='best')

    if xtag is None:
        xtag = 'Generation'

    if ytag is None:
        ytag = 'Fitness'

    axes.set_ylabel(ytag)
    axes.set_xlabel(xtag)

    fig.suptitle('Fitnesses of Agents Across Generations')
    fig.set_size_inches(7, 5)
    fig.set_dpi(100)

    # todo save fitness figure
    plt.show()
    return
