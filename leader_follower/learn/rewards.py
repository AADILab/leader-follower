import copy

import numpy as np

from leader_follower.leader_follower_env import LeaderFollowerEnv


def call_counter(func):
    def helper(*args, **kwargs):
        helper.n_calls += 1
        return func(*args, **kwargs)
    helper.n_calls = 0
    return helper

@call_counter
def calc_global(env: LeaderFollowerEnv):
    reward = env.num_poi_observed() / len(env.pois)
    return reward

def assign_followers(env):
    # todo change difference calculation to use this function
    assigned_followers = {
        leader_name: [leader_name]
        for leader_name, leader in env.leaders.items()
    }
    assigned_followers["leader_null"] = []

    follower_influences = {
        follower_name: follower.influence_counts()[0]
        for follower_name, follower in env.followers.items()
    }

    for follower_name, counts in follower_influences.items():
        for idx, name in enumerate(counts[0]):
            if not name.startswith('leader'):
                counts[1][idx] = -1
        if len(counts[1]) == 0:
            max_influencer = "leader_null"
        else:
            max_idx = np.argmax(counts[1])
            max_influencer = counts[0][max_idx]
        assigned_followers[max_influencer].append(follower_name)
    return assigned_followers


def calc_diff_rewards(env: LeaderFollowerEnv, remove_followers=False):
    """

    :param env:
    :param remove_followers:
    :return:
    """
    calc_global.n_calls = 0
    assigned_followers = {
        leader_name: []
        for leader_name, leader in env.leaders.items()
    }
    assigned_followers["leader_null"] = []

    if remove_followers:
        follower_influences = {
            follower_name: follower.influence_counts()[0]
            for follower_name, follower in env.followers.items()
        }

        for follower_name, counts in follower_influences.items():
            for idx, name in enumerate(counts[0]):
                if not name.startswith('leader'):
                    counts[1][idx] = -1
            if len(counts[1]) == 0:
                max_influencer = "leader_null"
            else:
                max_idx = np.argmax(counts[1])
                max_influencer = counts[0][max_idx]
            assigned_followers[max_influencer].append(follower_name)

    # todo  explore: if multiple agents are individually capable of observing a poi, neither receives a reward
    global_reward = calc_global(env)
    difference_rewards = {"G": global_reward}
    for leader, removed_agents in assigned_followers.items():
        removed_agents.append(leader)

        poi_copy = copy.deepcopy(env.pois)
        for poi_name, poi in env.pois.items():
            pruned_history = []
            for observation_step in poi.observation_history:
                pruned_step = [
                    agent
                    for agent in observation_step
                    if agent.name not in removed_agents
                ]
                pruned_history.append(pruned_step)
            poi.observation_history = pruned_history
        difference_global = calc_global(env)
        difference_rewards[leader] = global_reward - difference_global
        env.pois = poi_copy
    return difference_rewards, calc_global.n_calls

def calc_dpp_n(env: LeaderFollowerEnv, agent_names, n):
    """
    Calculate the reward in the counterfactual case where there are n copies of the given agent.
    This is equivalent to multiplying the agents true value by n.

    agent_names is an iterable and copy each agent in the iterable n times

    Note that when n = 0, this is effectively the difference reward.

    :param env:
    :param agent_names:
    :param n:
    :return:
    """
    orig_vals = {}
    for each_name in agent_names:
        agent = env.agent_mapping[each_name]
        orig_agent_val = agent.value
        agent.value = orig_agent_val * n
        orig_vals[each_name] = orig_agent_val

    # todo verify calculation of dpp_n
    reweighted_reward = calc_global(env)
    for each_name, each_orig_val in orig_vals.items():
        agent = env.agent_mapping[each_name]
        agent.value = each_orig_val
    return reweighted_reward

def calc_dpp(env: LeaderFollowerEnv, remove_followers=False):
    """
    Calculate D++ rewards for each rover

    DPP pseudocode

    1. calculate D++^{-1}
    2. calculate D++^{total_agents - 1}
    3. if D++^{total_agents - 1} <= D++^{-1}
    4.  return D++^{total_agents - 1}
    5. else:
    6.  n := 0
    7.  repeat:
    8.      n += 1
    9.      calculate D++^{n}
    10.     if calculate D++^{n} > D++^{n - 1}
    11.         return D++^{n}
    12. until n <= total_agents - 1
    13. return D++^{-1}

    :param env:
    :param remove_followers:
    :return dpp_rewards: Numpy array containing each rover's D++ reward
    """
    calc_global.n_calls = 0
    num_agents = len(env.leaders)
    dpp_rewards = {name: 0 for name, agent in env.leaders.items()}
    assigned = assign_followers(env)
    for leader_name, leader in env.leaders.items():
        # add assigning followers to the leader before calculating the dpp reward for the given agent
        agent_names = [leader_name]
        if remove_followers:
            follower_names = assigned[leader_name]
            agent_names.extend(follower_names)

        dpp_min = calc_dpp_n(env, agent_names=agent_names, n=0)
        dpp_max = calc_dpp_n(env, agent_names=agent_names, n=num_agents - 1)

        if dpp_max <= dpp_min:
            dpp_rewards[leader_name] = dpp_max
        else:
            dpp_rewards[leader_name] = dpp_min
            prev_dpp_n = dpp_min
            for val_n in range(1, num_agents - 1):
                dpp_n = calc_dpp_n(env, agent_names=[leader_name], n=val_n)
                if dpp_n > prev_dpp_n:
                    dpp_rewards[leader_name] = dpp_n
                    break
                prev_dpp_n = dpp_n
    return dpp_rewards, calc_global.n_calls
