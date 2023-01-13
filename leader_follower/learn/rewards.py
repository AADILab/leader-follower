import copy

import numpy as np


def calc_global_reward(poi_colony):
    # todo implement
    return 0

# def global_reward(poi_colony: POIColony):
#     return float(poi_colony.num_observed()) / float(poi_colony.num_pois)
#
#
# def individual_diff_reward(leader: Boid, follower_ids: list[int], poi_colony):
#     # Make a copy of the POI manager and all POIs
#     poi_colony_copy = copy.deepcopy(poi_colony)
#     all_removed_ids = follower_ids + [leader.id]
#     for poi in poi_colony_copy.pois:
#         # Determine if POI would be observed without this agent and its followers
#         # Set observation to False
#         poi.observed = False
#         # Check each group that observed this poi
#         for group in poi.observation_list:
#             # Recreate this group but with the leader and followers removed
#             # If the leaders and followers were not in this group, then this is just
#             # a copy of the original group
#             difference_group = [each_id for each_id in group if each_id not in all_removed_ids]
#             # If the coupling requirement is still satisfied, then set this poi as observed
#             if len(difference_group) >= poi_colony.coupling:
#                 poi.observed = True
#                 break
#     return global_reward(poi_colony) - global_reward(poi_colony_copy)
#
#
# def calc_diff_rewards(boids_colony, poi_colony):
#     # Assign followers to each leader
#     all_assigned_followers = [
#         []
#         for _ in range(boids_colony.bounds.num_leaders)
#     ]
#     for follower in boids_colony.getFollowers():
#         # Get the id of the max number in the influence list
#         # (this is the id of the leader that influenced this follower the most)
#         all_assigned_followers[argmax(follower.leader_influence)].append(follower.id)
#     difference_rewards = []
#     for leader, assigned_followers in zip(boids_colony.getLeaders(), all_assigned_followers):
#         difference_rewards.append(individual_diff_reward(leader, assigned_followers, poi_colony))
#     return difference_rewards

def calc_difference(pois, global_reward, rov_poi_dist, num_steps=500, observation_radius=1):
    """
    Calculate each rover's difference reward for the current episode
    :param pois: Dictionary containing POI class instances
    :param global_reward: Episodic global reward
    :param rov_poi_dist: Array containing distances between POI and rovers for entire episode
    :return difference_rewards: Numpy array containing each rover's difference reward
    """
    num_agents = len(rov_poi_dist)
    difference_rewards = np.zeros(num_agents)
    for agent_id in range(num_agents):
        counterfactual_global_reward = 0.0
        for pk in pois:  # For each POI
            poi_reward = 0.0  # Track best POI reward over all time steps for given POI
            for step in range(num_steps):
                observer_count = 0
                rover_distances = copy.deepcopy(rov_poi_dist[pois[pk].poi_id][step])
                rover_distances[agent_id] = 1000.00  # Replace Rover action with counterfactual action
                sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                # Check if required observers within range of POI
                for i in range(int(pois[pk].coupling)):
                    if sorted_distances[i] < observation_radius:
                        observer_count += 1

                # Calculate reward for given POI at current time step
                if observer_count >= int(pois[pk].coupling):
                    summed_observer_distances = sum(sorted_distances[0:int(pois[pk].coupling)])
                    reward = pois[pk].value/(summed_observer_distances/pois[pk].coupling)
                    if reward > poi_reward:
                        poi_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += poi_reward

        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards

def calc_dpp(pois, global_reward, rov_poi_dist, num_steps=500, observation_radius=1):
    """
    Calculate D++ rewards for each rover
    :param pois: Dictionary containing POI class instances
    :param global_reward: Episodic global reward
    :param rov_poi_dist: Array containing distances between POI and rovers for entire episode
    :return dpp_rewards: Numpy array containing each rover's D++ reward
    """
    num_agents = len(rov_poi_dist)
    d_rewards = calc_difference(pois, global_reward, rov_poi_dist)
    rewards = np.zeros(num_agents)  # This is just a temporary reward tracker for iterations of counterfactuals
    dpp_rewards = np.zeros(num_agents)

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    for agent_id in range(num_agents):
        counterfactual_global_reward = 0.0
        n_counters = num_agents - 1
        for pk in pois:
            poi_reward = 0.0  # Track best POI reward over all time steps for given POI
            for step in range(num_steps):
                observer_count = 0
                # print(rov_poi_dist[pois[pk].poi_id][step])
                rover_distances = copy.deepcopy(rov_poi_dist[pois[pk].poi_id][step])
                counterfactual_rovers = np.ones(int(n_counters)) * rover_distances[agent_id]
                rover_distances = np.append(rover_distances, counterfactual_rovers)
                sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                # Check if required observers within range of POI
                for i in range(int(pois[pk].coupling)):
                    if sorted_distances[i] < observation_radius:
                        observer_count += 1

                # Calculate reward for given POI at current time step
                if observer_count >= pois[pk].coupling:
                    summed_observer_distances = sum(sorted_distances[0:int(pois[pk].coupling)])
                    reward = pois[pk].value/(summed_observer_distances/pois[pk].coupling)
                    if reward > poi_reward:
                        poi_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += poi_reward

        rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters

    for agent_id in range(num_agents):
        # Compare D++ to D, and iterate through n counterfactuals if D++ > D
        if rewards[agent_id] > d_rewards[agent_id]:
            n_counters = 1
            while n_counters < num_agents:
                counterfactual_global_reward = 0.0
                for pk in pois:
                    observer_count = 0
                    poi_reward = 0.0  # Track best POI reward over all time steps for given POI
                    for step in range(num_steps):
                        rover_distances = copy.deepcopy(rov_poi_dist[pois[pk].poi_id][step])
                        counterfactual_rovers = np.ones(int(n_counters)) * rover_distances[agent_id]
                        rover_distances = np.append(rover_distances, counterfactual_rovers)
                        sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                        # Check if required observers within range of POI
                        for i in range(int(pois[pk].coupling)):
                            if sorted_distances[i] < observation_radius:
                                observer_count += 1

                        # Calculate reward for given POI at current time step
                        if observer_count >= pois[pk].coupling:
                            summed_observer_distances = sum(sorted_distances[0:int(pois[pk].coupling)])
                            reward = pois[pk].value/(summed_observer_distances/pois[pk].coupling)
                            if reward > poi_reward:
                                poi_reward = reward

                    # Update Counterfactual G
                    counterfactual_global_reward += poi_reward

                # Calculate D++ reward with n counterfactuals added
                temp_dpp = (counterfactual_global_reward - global_reward)/n_counters
                if temp_dpp > rewards[agent_id]:
                    rewards[agent_id] = temp_dpp
                    n_counters = num_agents + 1  # Stop iterating
                else:
                    n_counters += 1

            dpp_rewards[agent_id] = rewards[agent_id]  # Returns D++ reward for this agent
        else:
            dpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward for this agent

    return dpp_rewards
