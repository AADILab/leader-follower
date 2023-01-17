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


def calc_diff_rewards(env: LeaderFollowerEnv):
    """

    :param env:
    :return:
    """
    calc_global.n_calls = 0
    assigned_followers = {
        leader_name: []
        for leader_name, leader in env.leaders.items()
    }

    follower_influences = {
        follower_name: follower.influence_counts()[0]
        for follower_name, follower in env.followers.items()
    }

    for follower_name, counts in follower_influences.items():
        for idx, name in enumerate(counts[0]):
            if not name.startswith('leader'):
                counts[1][idx] = -1
        max_idx = np.argmax(counts[1])
        max_influencer = counts[0][max_idx]
        assigned_followers[max_influencer].append(follower_name)

    global_reward = calc_global(env)
    difference_rewards = {}
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
            largest_observation = max(pruned_history, key=len)
            poi.observed = len(largest_observation) >= poi.coupling
        difference_global = calc_global(env)
        difference_rewards[leader] = global_reward - difference_global
        env.pois = poi_copy
    return difference_rewards, calc_global.n_calls

def calc_dpp(env: LeaderFollowerEnv):
    """
    Calculate D++ rewards for each rover

    :param pois: Dictionary containing POI class instances
    :param global_reward: Episodic global reward
    :param rov_poi_dist: Array containing distances between POI and rovers for entire episode
    :return dpp_rewards: Numpy array containing each rover's D++ reward
    """
    # todo add tracking of calls to G
    # todo fix indexing pois
    # todo fix rov_poi_dist being part of the reward

    num_steps = 500
    d_rewards, g_calls = calc_diff_rewards(env)
    g_reward = calc_global(env)
    num_agents = len(env.leaders)

    # These are just temporary reward trackers for iterations of counterfactuals
    rewards = {name: 0 for name, agent in env.leaders.items()}
    dpp_rewards = {name: 0 for name, agent in env.leaders.items()}

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    for leader_name, leader in env.leaders.items():
        counterfactual_global_reward = 0.0
        n_counters = num_agents - 1
        for pk in env.pois:
            # Track best POI reward over all time steps for given POI
            poi_reward = 0.0
            for step in range(num_steps):
                observer_count = 0
                # print(rov_poi_dist[pois[pk].poi_id][step])
                rover_distances = copy.deepcopy(rov_poi_dist[pois[pk].poi_id][step])
                counterfactual_rovers = np.ones(int(n_counters)) * rover_distances[leader_name]
                rover_distances = np.append(rover_distances, counterfactual_rovers)
                sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                # Check if required observers within range of POI
                for i in range(int(env.pois[pk].coupling)):
                    if sorted_distances[i] < leader.observation_radius:
                        observer_count += 1

                # Calculate reward for given POI at current time step
                if observer_count >= env.pois[pk].coupling:
                    summed_observer_distances = sum(sorted_distances[0:int(env.pois[pk].coupling)])
                    reward = env.pois[pk].value/(summed_observer_distances/env.pois[pk].coupling)
                    if reward > poi_reward:
                        poi_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += poi_reward

        rewards[leader_name] = (counterfactual_global_reward - g_reward) / n_counters

    for leader_name, leader in env.leaders.items():
        # Compare D++ to D, and iterate through n counterfactuals if D++ > D
        if rewards[leader_name] <= d_rewards[leader_name]:
            dpp_rewards[leader_name] = d_rewards[leader_name]  # Returns difference reward for this agent
        else:
            n_counters = 1
            while n_counters < num_agents:
                counterfactual_global_reward = 0.0
                for pk in env.pois:
                    observer_count = 0
                    # Track best POI reward over all time steps for given POI
                    poi_reward = 0.0
                    for step in range(num_steps):
                        rover_distances = copy.deepcopy(rov_poi_dist[env.pois[pk].poi_id][step])
                        counterfactual_rovers = np.ones(int(n_counters)) * rover_distances[leader_name]
                        rover_distances = np.append(rover_distances, counterfactual_rovers)
                        sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                        # Check if required observers within range of POI
                        for i in range(int(env.pois[pk].coupling)):
                            if sorted_distances[i] < leader.observation_radius:
                                observer_count += 1

                        # Calculate reward for given POI at current time step
                        if observer_count >= env.pois[pk].coupling:
                            summed_observer_distances = sum(sorted_distances[0:int(env.pois[pk].coupling)])
                            reward = env.pois[pk].value/(summed_observer_distances/env.pois[pk].coupling)
                            if reward > poi_reward:
                                poi_reward = reward

                    # Update Counterfactual G
                    counterfactual_global_reward += poi_reward

                # Calculate D++ reward with n counterfactuals added
                temp_dpp = (counterfactual_global_reward - g_reward) / n_counters
                if temp_dpp > rewards[leader_name]:
                    rewards[leader_name] = temp_dpp
                    n_counters = num_agents + 1  # Stop iterating
                else:
                    n_counters += 1

            dpp_rewards[leader_name] = rewards[leader_name]  # Returns D++ reward for this agent

    return dpp_rewards

# def calc_difference(env: LeaderFollowerEnv):
# # def calc_difference(pois, global_reward, rov_poi_dist, num_steps=500, observation_radius=1):
#     """
#     Calculate each rover's difference reward for the current episode
#     :param pois: Dictionary containing POI class instances
#     :param global_reward: Episodic global reward
#     :param rov_poi_dist: Array containing distances between POI and rovers for entire episode
#     :return difference_rewards: Numpy array containing each rover's difference reward
#     """
#     # todo add tracking of calls to G
#     num_agents = len(rov_poi_dist)
#     difference_rewards = np.zeros(num_agents)
#     for agent_id in range(num_agents):
#         counterfactual_global_reward = 0.0
#         for pk in pois:  # For each POI
#             poi_reward = 0.0  # Track best POI reward over all time steps for given POI
#             for step in range(num_steps):
#                 observer_count = 0
#                 rover_distances = copy.deepcopy(rov_poi_dist[pois[pk].poi_id][step])
#                 rover_distances[agent_id] = 1000.00  # Replace Rover action with counterfactual action
#                 sorted_distances = np.sort(rover_distances)  # Sort from least to greatest
#
#                 # Check if required observers within range of POI
#                 for i in range(int(pois[pk].coupling)):
#                     if sorted_distances[i] < observation_radius:
#                         observer_count += 1
#
#                 # Calculate reward for given POI at current time step
#                 if observer_count >= int(pois[pk].coupling):
#                     summed_observer_distances = sum(sorted_distances[0:int(pois[pk].coupling)])
#                     reward = pois[pk].value/(summed_observer_distances/pois[pk].coupling)
#                     if reward > poi_reward:
#                         poi_reward = reward
#
#             # Update Counterfactual G
#             counterfactual_global_reward += poi_reward
#
#         difference_rewards[agent_id] = global_reward - counterfactual_global_reward
#
#     return difference_rewards
