import copy

import numpy as np

from leader_follower.leader_follower_env import LeaderFollowerEnv


def calc_global(env: LeaderFollowerEnv):
    reward = env.num_poi_observed() / len(env._pois)
    return reward


def calc_diff_rewards(env: LeaderFollowerEnv):
    """

    :param env:
    :return:
    """
    assigned_followers = {
        leader_name: []
        for leader_name, leader in env._leaders.items()
    }

    assigned_followers["leader_null"] = []

    follower_influences = {
        follower_name: follower.influence_counts()[0]
        for follower_name, follower in env._followers.items()
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

    global_reward = calc_global(env)
    difference_rewards = {"G": global_reward}
    for leader, removed_agents in assigned_followers.items():
        removed_agents.append(leader)

        poi_copy = copy.deepcopy(env._pois)
        for poi_name, poi in env._pois.items():
            pruned_history = []
            for observation_step in poi.observation_history:
                pruned_step = [
                    agent
                    for agent in observation_step
                    if agent.name not in removed_agents
                ]
                pruned_history.append(pruned_step)
            if len(pruned_history) == 0:
                largest_observation = []
            else:
                largest_observation = max(pruned_history, key=len)
            poi.observed = len(largest_observation) >= poi.coupling
        difference_global = calc_global(env)
        difference_rewards[leader] = global_reward - difference_global
        env._pois = poi_copy
    # print(global_reward, difference_rewards)
    return difference_rewards

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
#
# def calc_dpp(env: LeaderFollowerEnv):
# # def calc_dpp(pois, global_reward, rov_poi_dist, num_steps=500, observation_radius=1):
#     """
#     Calculate D++ rewards for each rover
#     :param pois: Dictionary containing POI class instances
#     :param global_reward: Episodic global reward
#     :param rov_poi_dist: Array containing distances between POI and rovers for entire episode
#     :return dpp_rewards: Numpy array containing each rover's D++ reward
#     """
#     # todo add tracking of calls to G
#     num_agents = len(rov_poi_dist)
#     d_rewards = calc_difference(pois, global_reward, rov_poi_dist)
#     rewards = np.zeros(num_agents)  # This is just a temporary reward tracker for iterations of counterfactuals
#     dpp_rewards = np.zeros(num_agents)
#
#     # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
#     for agent_id in range(num_agents):
#         counterfactual_global_reward = 0.0
#         n_counters = num_agents - 1
#         for pk in pois:
#             poi_reward = 0.0  # Track best POI reward over all time steps for given POI
#             for step in range(num_steps):
#                 observer_count = 0
#                 # print(rov_poi_dist[pois[pk].poi_id][step])
#                 rover_distances = copy.deepcopy(rov_poi_dist[pois[pk].poi_id][step])
#                 counterfactual_rovers = np.ones(int(n_counters)) * rover_distances[agent_id]
#                 rover_distances = np.append(rover_distances, counterfactual_rovers)
#                 sorted_distances = np.sort(rover_distances)  # Sort from least to greatest
#
#                 # Check if required observers within range of POI
#                 for i in range(int(pois[pk].coupling)):
#                     if sorted_distances[i] < observation_radius:
#                         observer_count += 1
#
#                 # Calculate reward for given POI at current time step
#                 if observer_count >= pois[pk].coupling:
#                     summed_observer_distances = sum(sorted_distances[0:int(pois[pk].coupling)])
#                     reward = pois[pk].value/(summed_observer_distances/pois[pk].coupling)
#                     if reward > poi_reward:
#                         poi_reward = reward
#
#             # Update Counterfactual G
#             counterfactual_global_reward += poi_reward
#
#         rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters
#
#     for agent_id in range(num_agents):
#         # Compare D++ to D, and iterate through n counterfactuals if D++ > D
#         if rewards[agent_id] > d_rewards[agent_id]:
#             n_counters = 1
#             while n_counters < num_agents:
#                 counterfactual_global_reward = 0.0
#                 for pk in pois:
#                     observer_count = 0
#                     poi_reward = 0.0  # Track best POI reward over all time steps for given POI
#                     for step in range(num_steps):
#                         rover_distances = copy.deepcopy(rov_poi_dist[pois[pk].poi_id][step])
#                         counterfactual_rovers = np.ones(int(n_counters)) * rover_distances[agent_id]
#                         rover_distances = np.append(rover_distances, counterfactual_rovers)
#                         sorted_distances = np.sort(rover_distances)  # Sort from least to greatest
#
#                         # Check if required observers within range of POI
#                         for i in range(int(pois[pk].coupling)):
#                             if sorted_distances[i] < observation_radius:
#                                 observer_count += 1
#
#                         # Calculate reward for given POI at current time step
#                         if observer_count >= pois[pk].coupling:
#                             summed_observer_distances = sum(sorted_distances[0:int(pois[pk].coupling)])
#                             reward = pois[pk].value/(summed_observer_distances/pois[pk].coupling)
#                             if reward > poi_reward:
#                                 poi_reward = reward
#
#                     # Update Counterfactual G
#                     counterfactual_global_reward += poi_reward
#
#                 # Calculate D++ reward with n counterfactuals added
#                 temp_dpp = (counterfactual_global_reward - global_reward)/n_counters
#                 if temp_dpp > rewards[agent_id]:
#                     rewards[agent_id] = temp_dpp
#                     n_counters = num_agents + 1  # Stop iterating
#                 else:
#                     n_counters += 1
#
#             dpp_rewards[agent_id] = rewards[agent_id]  # Returns D++ reward for this agent
#         else:
#             dpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward for this agent
#
#     return dpp_rewards
