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

def rollout(env, individuals, reward_type=None, render=False):
    networks: list[NeuralNetwork] = [ind['network'] for ind in individuals]

    env.reset()
    done = env.done()
    global_reward = env.global_reward()

    while not done:
        state = env.state()
        next_actions = [get_action(each_net, state, env) for each_net in networks]
        next_state, agent_reward, global_reward, done = env.step(next_actions, render=False)
        if render:
            env.display()

    episode_reward = [global_reward for _ in env.agents]
    # episode_reward = env.agent_episode_rewards()
    # if reward_type == 'global':
    #     episode_reward = [global_reward for _ in episode_reward]
    return episode_reward


def downselect_top_n(population, max_size):
    sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
    return sorted_pop[:max_size]


def neuro_evolve(env, n_hidden, population_size, n_gens, sim_pop_size, reward_type=None):
    debug = False
    select_func = select_roulette
    mutate_func = mutate_gaussian
    downselect_func = downselect_top_n

    # only creat sub-pops for agents capable of learning
    agent_pops = [
        [
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
    ]
    print(f'Using device: {agent_pops[0][0]["network"].device()}')

    # initial fitness evaluation of all networks in population
    for pop_idx in range(population_size):
        new_inds = [agent_pops[agent_idx][pop_idx] for agent_idx in range(len(agent_pops))]
        agent_rewards = rollout(env, new_inds, reward_type=reward_type, render=debug)
        for each_ind, each_fitness in zip(new_inds, agent_rewards):
            each_ind['fitness'] = each_fitness

    max_fitnesses = []
    avg_fitnesses = []
    for gen_idx in trange(n_gens):
        fitnesses = [
            [ind['fitness'] for ind in each_pop]
            for each_pop in agent_pops
        ]
        max_fitnesses.append(np.max(fitnesses, axis=1))
        avg_fitnesses.append(np.average(fitnesses, axis=1))
        sim_pops = [select_func(pop, sim_pop_size) for pop in agent_pops]
        for sim_pop_idx, each_ind in enumerate(sim_pops):
            new_inds = [mutate_func(agent_pops[agent_idx][sim_pop_idx]) for agent_idx in range(len(agent_pops))]

            # rollout and evaluate
            agent_rewards = rollout(env, new_inds, reward_type=reward_type, render=debug)
            for each_agent, each_fitness in zip(new_inds, agent_rewards):
                each_agent['fitness'] = each_fitness

            # reinsert
            for each_entire, each_new in zip(agent_pops, new_inds):
                each_entire.append(each_new)
        # downselect
        agent_pops = [downselect_func(pop, population_size) for pop in agent_pops]

    best_policies = []
    for each_pop in agent_pops:
        fitness_vals = [pop['fitness'] for pop in each_pop]
        arg_best = np.argmax(fitness_vals)
        best_ind = each_pop[arg_best]
        print(best_ind['fitness'])
        best_policies.append(best_ind)
    return best_policies, max_fitnesses, avg_fitnesses


def plot_fitnesses(avg_fitnesses, max_fitnesses, xtag=None, ytag=None):
    fig, axes = plt.subplots()

    avg_fitnesses = np.transpose(avg_fitnesses)
    max_fitnesses = np.transpose(max_fitnesses)

    for idx, each_fitness in enumerate(avg_fitnesses):
        axes.plot(each_fitness, label=f'Avg: Agent {idx}')

    for idx, each_fitness in enumerate(max_fitnesses):
        axes.plot(each_fitness, label=f'Max: Agent {idx}')

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
