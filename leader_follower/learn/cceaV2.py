"""
@title

@description

"""
import copy
from functools import partial
from pathlib import Path
from threading import Thread

import numpy as np
import pandas as pd
import torch
from numpy.random import default_rng
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import trange

from leader_follower.agent import AgentType
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork
from leader_follower.learn.rewards import calc_global


def select_roulette(agent_pops, select_size, noise=0.01):
    """
    output a list of dicts, where each idx contains the policy for each agent
    :param agent_pops:
    :param select_size:
    :param noise:
    :return:
    """
    chosen_agent_pops = {}
    for agent_name, policy_population in agent_pops.items():
        pop_copy = [
            {'network': each_policy['network'].copy(), 'fitness': each_policy['fitness']}
            for each_policy in policy_population
        ]
        fitness_vals = np.asarray([pop['fitness'] for pop in pop_copy])

        # add small amount of noise to each fitness value (help deal with all same value)
        noise = np.random.uniform(0, noise, len(fitness_vals))
        fitness_vals += noise

        # adding an extra bogey population and fitness value fixes an issue arising when trying to select from the
        # population when select_size == len(population), which causes the min value of the normalized fitnesses
        # to be 0, which is not a valid probability. The extra fitness makes it so the 0 probability fitness is not
        # an actual fitness, and we have to add an extra (fake) population to make the lengths of the population
        # and fitness arrays to be equal. They are both added at the end of the array, so the fake population
        # is correlated with the fake fitness.
        min_fitness = np.min(fitness_vals)
        fitness_vals = np.append(fitness_vals, min_fitness - 2)

        bogey_entry = {'network': None, 'fitness': -1}
        pop_copy.append(bogey_entry)

        # if there is one positive fitness, then the others all being negative causes issues when calculating the probs
        fitness_norms = (fitness_vals - np.min(fitness_vals)) / (np.max(fitness_vals) - np.min(fitness_vals))
        if np.isnan(fitness_norms).any():
            fitness_norms = fitness_vals
        fitness_probs = fitness_norms / np.sum(fitness_norms)

        # validation check to make sure adding the bogey fitness and policy do not mess up the random selection
        arg_minfitness = np.argmin(fitness_probs)
        assert arg_minfitness == fitness_probs.size - 1
        assert pop_copy[arg_minfitness] == bogey_entry

        rand_pop = np.random.choice(pop_copy, select_size, replace=False, p=fitness_probs)
        chosen_agent_pops[agent_name] = rand_pop

    chosen_pops = [
        {agent_name: pops[idx] for agent_name, pops in chosen_agent_pops.items()}
        for idx in range(0, select_size)
    ]
    return chosen_pops

def select_egreedy(agent_pops, select_size, epsilon):
    # todo implement egreedy selection
    rng = default_rng()
    chosen_agent_pops = {}
    for agent_name, policy_population in agent_pops.items():
        rand_val = rng.random()
        if rand_val <= epsilon:
            pass
        else:
            pass
        # chosen_agent_pops[agent_name] = rand_pop

    chosen_pops = [
        {agent_name: pops[idx] for agent_name, pops in chosen_agent_pops.items()}
        for idx in range(0, select_size)
    ]
    return chosen_pops

def select_hall_of_fame(agent_pops):
    # todo implement hall of fame selection
    return

def select_top_n(agent_pops, select_size):
    chosen_agent_pops = {}
    for agent_name, population in agent_pops.items():
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
        top_pop = sorted_pop[:select_size]
        chosen_agent_pops[agent_name] = top_pop

    chosen_pops = [
        {agent_name: pops[idx] for agent_name, pops in chosen_agent_pops.items()}
        for idx in range(0, select_size)
    ]
    return chosen_pops


def mutate_gaussian(agent_policies, proportion=0.1, probability=0.05):
    mutated_agents = {}
    for agent_name, individual in agent_policies.items():
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
        mutated_agents[agent_name] = new_ind
    return mutated_agents


def downselect_top_n(agent_pops, select_size):
    chosen_agent_pops = {}
    for agent_name, population in agent_pops.items():
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
        top_pop = sorted_pop[:select_size]
        chosen_agent_pops[agent_name] = top_pop
    return chosen_agent_pops

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

    episode_rewards = reward_func(env)
    if isinstance(episode_rewards, float) or isinstance(episode_rewards, int):
        episode_rewards = {name: episode_rewards for name in env.leaders.keys()}
    return episode_rewards

def simulate_subpop(env, agent_policies, mutate_func, reward_func):
    agent_policies = mutate_func(agent_policies)

    # rollout and evaluate
    agent_rewards = rollout(env, agent_policies, reward_func=reward_func, render=False)
    for agent_name, policy_info in agent_policies.items():
        policy_fitness = agent_rewards[agent_name]
        policy_info['fitness'] = policy_fitness
        # reinsert new individual into population of policies
        # simulated_policies[agent_name] = policy_info
    return agent_policies

def save_agent_policies(experiment_dir, gen_idx, env, agent_pops, fitnesses):
    gen_path = Path(experiment_dir, f'gen_{gen_idx}')
    if not gen_path:
        gen_path.mkdir(parents=True, exist_ok=True)

    # env_save_path = Path(gen_path, 'envs')
    # if not env_save_path:
    #     env_save_path.mkdir(parents=True, exist_ok=True)

    for agent_name, policy_info in agent_pops.items():
        network_save_path = Path(gen_path, f'{agent_name}_networks')
        if not network_save_path:
            network_save_path.mkdir(parents=True, exist_ok=True)

        for idx, each_policy in enumerate(policy_info):
            fitnesses[agent_name].append(each_policy['fitness'])
            network = each_policy['network']
            network.save_model(save_dir=network_save_path, tag=idx)

    env.save_environment(gen_path, tag=f'gen_{gen_idx}')

    fitnesses_path = Path(gen_path, 'fitnesses.csv')
    fitnesses_df = pd.DataFrame.from_dict(fitnesses, orient='index')
    fitnesses_df.to_csv(fitnesses_path, header=True, index_label='agent_name')
    return

def neuro_evolve(env: LeaderFollowerEnv, n_hidden, population_size, n_gens, sim_pop_size, reward_func, experiment_dir):
    # todo  implement leniency
    debug = False
    select_func = partial(select_roulette, **{'select_size': sim_pop_size, 'noise': 0.01})
    mutate_func = partial(mutate_gaussian, proportion=0.1, probability=0.05)
    downselect_func = partial(downselect_top_n, **{'select_size': population_size})

    # todo  allow for policy sharing
    #       move set of policies into agent definition add select policy based on idx
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
        if env.agent_mapping[agent_name].type == AgentType.Learner
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

    saving_threads: list[Thread] = []
    env.save_environment(experiment_dir, tag='initial')
    for gen_idx in trange(n_gens):
        sim_pops = select_func(agent_pops)
        # todo  multiprocess simulating each simulation population
        for agent_policies in sim_pops:
            # todo check to make sure previous policy isn't being overwritten
            # env, agent_policies, mutate_func, reward_func
            agent_policies = simulate_subpop(env, agent_policies, mutate_func, reward_func)
            for name, policy in agent_policies.items():
                agent_pops[name].append(policy)

        #     agent_policies = mutate_func(agent_policies)
        #     # rollout and evaluate
        #     agent_rewards = rollout(env, agent_policies, reward_func=reward_func, render=debug)
        #     for agent_name, policy_info in agent_policies.items():
        #         policy_fitness = agent_rewards[agent_name]
        #         policy_info['fitness'] = policy_fitness
        #         # reinsert new individual into population of policies
        #         agent_pops[agent_name].append(policy_info)

        # downselect
        agent_pops = downselect_func(agent_pops)

        top_inds = select_top_n(agent_pops, select_size=1)[0]
        _ = rollout(env, top_inds, reward_func=reward_func, render=debug)
        g_reward = calc_global(env)
        fitnesses = {
            agent_name: []
            for agent_name, policy_info in agent_pops.items()
        }
        fitnesses['G'] = [g_reward for _ in range(0, population_size)]

        # save all policies of each agent
        # save fitnesses mapping policies to fitnesses
        save_agent_policies(experiment_dir, gen_idx, env, agent_pops, fitnesses)

        # experiment_dir, gen_idx, env, agent_pops, fitnesses
        # save_thread = Thread(target=save_agent_policies, args=(experiment_dir, gen_idx, env, agent_pops, fitnesses))
        # save_thread.start()
        # saving_threads.append(save_thread)

    for each_thread in saving_threads:
        each_thread.join()

    # todo better way of combining best policies from each population for a final solution
    top_inds = select_top_n(agent_pops, select_size=1)[0]
    return top_inds
