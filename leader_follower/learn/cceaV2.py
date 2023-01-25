"""
@title

@description

"""

import copy
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.random import default_rng
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import trange

from leader_follower import project_properties
from leader_follower.agent import AgentType
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork


def select_roulette(population, select_size):
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


def mutate_gaussian(individual, proportion=0.1, probability=0.05):
    model = individual['network']
    model_copy = copy.deepcopy(model)

    rng = default_rng()
    with torch.no_grad():
        param_vector = parameters_to_vector(model_copy.parameters())

        for each_val in param_vector:
            rand_val = rng.random()
            if rand_val <= probability:
                # todo  base proportion on current weight rather than scaled random sample
                noise = torch.randn(each_val.size()) * proportion
                each_val.add_(noise)

        vector_to_parameters(param_vector, model_copy.parameters())
    new_ind = {
        'network': model_copy,
        'fitness': None
    }
    return new_ind

def rollout(env: LeaderFollowerEnv, individuals, reward_func, render=False):
    observations = env.reset()
    agent_dones = env.done()
    done = all(agent_dones.values())

    # select policy to use for each learning agent
    for agent_name, policy_info in individuals.items():
        env.agent_mapping[agent_name].policy = policy_info['network']

    while not done:
        next_actions = env.get_actions_from_observations(observations=observations)
        observations, rewards, agent_dones, truncs, infos = env.step(next_actions)
        done = all(agent_dones.values())
        if render:
            env.render()

    # todo ask Ever about what issue this solved
    # if type(episode_rewards) == float:
    #     # agent_rewards = [agent_rewards] * (len(env._leaders)+len(env._followers))
    #     episode_rewards = {
    #         agent_name: episode_rewards for agent_name, _ in individuals.items()
    #     }
    #
    # return episode_rewards

    episode_rewards, g_calls = reward_func(env)
    return episode_rewards, g_calls


def downselect_top_n(population, max_size):
    sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
    return sorted_pop[:max_size]


def neuro_evolve(env: LeaderFollowerEnv, n_hidden, population_size, n_gens, sim_pop_size, reward_func):
    debug = False
    # todo  implement hall of fame
    # todo  implement leniency
    select_func = select_roulette
    mutate_func = partial(mutate_gaussian, proportion=0.1, probability=0.05)
    downselect_func = downselect_top_n

    # only creat sub-pops for agents capable of learning
    # todo  allow for policy sharing
    #       move set of policies into agent definition add select policy based on idx
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
        if env.agent_mapping[agent_name].type == AgentType.Learner
    }
    print(f'Using device: {list(agent_pops.values())[0][0]["network"].device()}')

    # initial fitness evaluation of all networks in population
    # todo check reward structure/assignment to make sure reward is being properly assigned
    for pop_idx in range(population_size):
        new_inds = {agent_name: policy_info[pop_idx] for agent_name, policy_info in agent_pops.items()}
        agent_rewards, g_calls = rollout(env, new_inds, reward_func=reward_func, render=debug)
        for agent_name, policy_info in agent_pops.items():
            policy_fitness = agent_rewards[agent_name]
            policy_info[pop_idx]['fitness'] = policy_fitness

    now = datetime.now()
    experiment_id = f'{now.strftime("%Y_%m_%d_%H_%M")}'
    gens_path = Path(project_properties.cached_dir, f'{experiment_id}_neuro_evolve', 'gens')
    if not gens_path:
        gens_path.mkdir(parents=True, exist_ok=True)

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

        env_save_path = Path(gens_path, 'envs')
        if not env_save_path:
            env_save_path.mkdir(parents=True, exist_ok=True)

        # todo  multiprocess simulating each simulation population
        for sim_pop_idx, each_ind in enumerate(sim_pops):
            new_inds = {
                agent_name: mutate_func(policy_info[sim_pop_idx])
                for agent_name, policy_info in agent_pops.items()
            }

            # rollout and evaluate
            agent_rewards, g_calls = rollout(env, new_inds, reward_func=reward_func, render=debug)
            for agent_name, policy_info in new_inds.items():
                policy_fitness = agent_rewards[agent_name]
                policy_info['fitness'] = policy_fitness
                # reinsert new individual into population of policies
                agent_pops[agent_name].append(policy_info)

            # save current state of env at each rollout (agent trajectories)
            env.save_environment(env_save_path, sim_pop_idx)

        # downselect
        agent_pops = {
            agent_name: downselect_func(policy_info, population_size)
            for agent_name, policy_info in agent_pops.items()
        }

        fitnesses = {
            agent_name: []
            for agent_name, policy_info in agent_pops.items()
        }
        for agent_name, policy_info in agent_pops.items():
            network_save_path = Path(gens_path, f'{agent_name}_networks')
            if not env_save_path:
                network_save_path.mkdir(parents=True, exist_ok=True)

            for idx, each_policy in enumerate(policy_info):
                fitnesses[agent_name].append(each_policy['fitness'])
                # save all policies of each agent
                network = each_policy['network']
                network.save_model(save_dir=network_save_path, tag=idx)

        # save fitnesses mapping policies to fitnesses
        fitnesses_path = Path(gens_path, 'fitnesses.csv')
        fitnesses_df = pd.DataFrame.from_dict(fitnesses, orient='index')
        fitnesses_df.to_csv(fitnesses_path, header=True, index_label='agent_name')

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

    # todo filter out non-learning agents
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

    fig_dir = Path(project_properties.output_dir, 'figs')
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True, exist_ok=True)

    num_figs = len(list(fig_dir.iterdir()))
    figname = Path(fig_dir, f'fitnesses_{num_figs}.png')
    plt.savefig(figname)
    return
