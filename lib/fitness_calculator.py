from typing import List, Optional, Union
from copy import deepcopy
from enum import IntEnum

import numpy as np

from lib.poi_colony import POIColony, POI
from lib.boids_colony import BoidsColony, Boid
from lib.math_helpers import argmax
from lib.np_helpers import invertInds

class WhichG(IntEnum):
    # Continuous is what D++ used. 
    # Inverse distance to POIs w. coupling through an episode
    Continuous = 0
    # MinContinous is continuous, but we only use the min inverse distance(s) at
    # a particular timestep. Found by iterating through each timestep in an episode
    MinContinuous = 1
    # MinDiscrete is the all or nothing reward structure. Basically, for each POI,
    # check if enough agents were within its observation radius at each point during the episode
    # If at any point, the coupling requirement is met, then we count that POI as observed
    # Num poi observed / total num poi
    MinDiscrete = 2

class WhichD(IntEnum):
    # Don't calculate a difference reward. Just give each agent G
    G = 0
    # Use standard D where we just remove this agent's trajectory
    D = 1
    # Calculate D of this (leader) agent and its followers
    DFollow = 2

class FitnessCalculator():
    def __init__(self, poi_colony: POIColony, boids_colony: BoidsColony, which_G: Union[WhichG, str], which_D: Union[WhichD, str]) -> None:
        self.poi_colony = poi_colony
        self.boids_colony = boids_colony
        
        if type(which_G) == str:
            which_G = WhichG[which_G]
        if type(which_D) == str:
            which_D = WhichD[which_D]
        
        self.which_G = which_G
        self.which_D = which_D

    def calculateG(self, position_history: List[np.ndarray]):
        if self.which_G == WhichG.Continuous:
            raise NotImplementedError()
        elif self.which_G == WhichG.MinContinuous:
            return self.calculateMinContinuousG(position_history)
        elif self.which_G == WhichG.MinDiscrete:
            return self.calculateMinDiscreteG(position_history)
    
    def calculateDs(self, G: float, position_history: List[np.ndarray]):
        if self.which_D == WhichD.G:
            return [G for leader in self.boids_colony.getLeaders()]
    
        elif self.which_D == WhichD.D:
            difference_evaluations = []
            for leader in self.boids_colony.getLeaders():
                D = self.calculateD(G, [leader.id], position_history)
                difference_evaluations.append(D)
            return difference_evaluations

        elif self.which_D == WhichD.DFollow:
            # Assign followers to each leader
            all_assigned_followers = [[] for _ in range(self.boids_colony.bounds.num_leaders)]
            for follower in self.boids_colony.getFollowers():
                # Get the id of the max number in the influence list (this is the id of the leader that influenced this follower the most)
                all_assigned_followers[argmax(follower.leader_influence)].append(follower.id)

            difference_follower_evaluations = []
            for leader in self.boids_colony.getLeaders():
                # Figure out which trajectories we're actually removing
                ids_to_remove = [leader.id]+all_assigned_followers[leader.id]
                # Calculate Dfollow
                D_follow = self.calculateD(G, ids_to_remove, position_history)
                difference_follower_evaluations.append(D_follow)
            return difference_follower_evaluations
            
    def calculateD(self, G: float, ids_to_remove: int, position_history: List[np.ndarray]):
        G_c = self.calculateCounterfactualG(position_history, ids_to_remove)
        if G_c > G:
            raise Exception("G_c > G | Counterfactual G was greater than actual G. Something went wrong calculating the counterfactual")
        return G-G_c
    
    def calculateDistances(self, poi, agent_positions):
        # Calculate the distances for one timestep for a given poi and positions of all agents
        return np.linalg.norm(poi.position - agent_positions, axis=1)

    def calculatePOIScore(self, poi: POI, position_history: List[np.ndarray]):
        # Go through position history. Calculate score for each timestep. Keep the highest one. 
        # Basically, the highest value observation is the one we count
        # Coupling determines how many agents we look at to determine the observation value
        highest_poi_score = 0
        for t, agent_positions in enumerate(position_history):
            distances = self.calculateDistances(poi, agent_positions)
            # Coupling determines how many distances we use here
            distances_sorted = np.sort(distances)
            # print("distances_sorted: ", distances_sorted)
            if len(distances_sorted) == 0:
                poi_score = 0
            elif len(distances_sorted) < self.poi_colony.coupling:
                poi_score = 0
            else:
                poi_score = 1/( (1/self.poi_colony.coupling) * (np.sum(distances_sorted[:self.poi_colony.coupling])) )
            # print(poi_score)
            if poi_score > highest_poi_score:
                highest_poi_score = poi_score
        return highest_poi_score

    def calculateMinContinuousG(self, position_history: List[np.ndarray]):
        # print("calculateMinContinuousG")
        # print(position_history)
        # if poi_colony is None:
        #     poi_colony = self.poi_colony
        total_score = 0
        for poi in self.poi_colony.pois:
            highest_poi_score = self.calculatePOIScore(poi, position_history)
            # Cap the highest score at 1.0 so agents can't get near infinite score for getting really close
            if highest_poi_score > 1.0:
                highest_poi_score = 1.0
            total_score += highest_poi_score
        return total_score
    
    def calculateMinDiscreteG(self, position_history: List[np.ndarray]):
        num_observed = 0.0
        for poi in self.poi_colony.pois:
            for t, agent_positions in enumerate(position_history):
                distances = self.calculateDistances(poi, agent_positions)
                # Figure out how many of these distances are within the observation radius
                within_obs_radius = 0
                for dist in distances:
                    if dist <= self.poi_colony.observation_radius:
                        within_obs_radius += 1
                # Check if enough were within the observation radius to satisfy the coupling requirement
                if within_obs_radius >= self.poi_colony.coupling:
                    num_observed += 1
                    # Break the for loop for this POI and move onto the next POI
                    # There are no double points for observing the same POI multiple times
                    break
        return num_observed/self.poi_colony.num_pois                
    
    def calculateCounterfactualG(self, position_history, ids_to_remove):
        # First just copy the position_history as a np array. Np array makes it easier to slice
        counterfactual_position_history = np.array(position_history).copy()
        # Invert ids to remove to which ones we're keeping
        ids_to_keep = invertInds(counterfactual_position_history.shape[1], ids_to_remove)
        # Calculate G but with those ids removed. Hence, G_c
        return self.calculateG(counterfactual_position_history[:,ids_to_keep,:])

    # def calculateCounterfactualTeamFitness(self, boid_id: int, position_history: List[np.ndarray]):
    #     # Remove the specified boid's trajectory from the position history and calculate the fitness

    #     # First just copy the position_history as a np array. Np array makes it easier to slice
    #     counterfactual_position_history = np.array(position_history).copy()
    #     # Array is (timesteps, agents, xy)
    #     counterfactual_position_history = np.concatenate(
    #         (counterfactual_position_history[:,:boid_id,:],
    #         counterfactual_position_history[:,boid_id+1:,:]),
    #         axis=1
    #     )
    #     # Special case where there is only one agent and removing it causes
    #     # the counterfactual position history to be empty
    #     if counterfactual_position_history.shape[1] == 0:
    #         return 0.0
    #     else:
    #         return self.calculateContinuousTeamFitness(poi_colony=None, position_history=counterfactual_position_history)
    
    # def calculateDifferenceEvaluations(self, position_history=List[np.ndarray]):
    #     G = self.calculateContinuousTeamFitness(poi_colony=None, position_history=position_history)
    #     difference_evaluations = []
    #     for leader in self.boids_colony.getLeaders():
    #         G_c = self.calculateCounterfactualTeamFitness(leader.id, position_history)
    #         D = G-G_c
    #         difference_evaluations.append(D)
    #     return difference_evaluations

    # def calculateDifferenceFollowerEvaluations(self, position_history=List[np.ndarray]):
    #     G = self.calculateContinuousTeamFitness(poi_colony=None, position_history=position_history)
    #     # Assign followers to each leader
    #     all_assigned_followers = [[] for _ in range(self.boids_colony.bounds.num_leaders)]
    #     for follower in self.boids_colony.getFollowers():
    #         # Get the id of the max number in the influence list (this is the id of the leader that influenced this follower the most)
    #         all_assigned_followers[argmax(follower.leader_influence)].append(follower.id)

    #     # First just copy the position_history as a np array. Np array makes it easier to slice
    #     counterfactual_position_history = np.array(position_history).copy()
        
    #     # Calculate the actual difference evaluations
    #     difference_follower_evaluations = []
    #     for leader in self.boids_colony.getLeaders():
    #         # Figure out which trajectories we're actually removing
    #         ids_to_remove = [leader.id]+all_assigned_followers[leader.id]
    #         # And from that, which ones we're keeping
    #         ids_to_keep = invertInds(counterfactual_position_history.shape[1], ids_to_remove)

    #         # Calculate G with only the trajectories we're keeping
    #         # (Remove leader and its followers)
    #         G_c = self.calculateContinuousTeamFitness(poi_colony=None, position_history=counterfactual_position_history[:,ids_to_keep,:])
    #         D_follow = G-G_c
    #         difference_follower_evaluations.append(D_follow)

    #     return difference_follower_evaluations

    # def calculateContinuousDifferenceFitness(self, leader: Boid, position_history: List[np.ndarray]):
    #     # Make a copy of the POI m

    # def getTeamFitness(self, poi_colony: Optional[POIColony] = None):
    #     if poi_colony is None:
    #         poi_colony = self.poi_colony
    #     return float(poi_colony.numObserved())/float(poi_colony.num_pois)

    # def calculateDifferenceEvaluation(self, leader: Boid, assigned_follower_ids: List[int]):
    #     # Make a copy of the POI manager and all POIs
    #     poi_colony_copy = deepcopy(self.poi_colony)
    #     all_removed_ids = assigned_follower_ids+[leader.id]
    #     for poi in poi_colony_copy.pois:
    #         # Determine if POI would be observed without this agent and its followers
    #         # Set observation to False
    #         poi.observed = False
    #         # Check each group that observed this poi
    #         for group in poi.observation_list:
    #             # Recreate this group but with the leader and followers removed
    #             # If the leaders and followers were not in this group, then this is just
    #             # a copy of the original group
    #             difference_group = [id for id in group if id not in all_removed_ids]
    #             # If the coupling requirement is still satisfied, then set this poi as observed
    #             if len(difference_group) >= self.poi_colony.coupling:
    #                 poi.observed = True
    #                 break
    #     return self.getTeamFitness() - self.getTeamFitness(poi_colony_copy)

    # def calculateDifferenceEvaluations(self):
    #     # Assign followers to each leader
    #     all_assigned_followers = [[] for _ in range(self.boids_colony.bounds.num_leaders)]
    #     for follower in self.boids_colony.getFollowers():
    #         # Get the id of the max number in the influence list (this is the id of the leader that influenced this follower the most)
    #         all_assigned_followers[argmax(follower.leader_influence)].append(follower.id)
    #     difference_rewards = []
    #     for leader, assigned_followers in zip(self.boids_colony.getLeaders(), all_assigned_followers):
    #         difference_rewards.append(self.calculateDifferenceEvaluation(leader, assigned_followers))
    #     return difference_rewards
