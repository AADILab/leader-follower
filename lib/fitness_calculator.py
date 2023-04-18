from typing import List, Optional
from copy import deepcopy

import numpy as np

from lib.poi_colony import POIColony, POI
from lib.boids_colony import BoidsColony, Boid
from lib.math_helpers import argmax

class FitnessCalculator():
    def __init__(self, poi_colony: POIColony, boids_colony: BoidsColony) -> None:
        self.poi_colony = poi_colony
        self.boids_colony = boids_colony
    
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
            poi_score = 1/( (1/self.poi_colony.coupling) * (np.sum(distances_sorted[:self.poi_colony.coupling])) )
            if poi_score > highest_poi_score:
                highest_poi_score = poi_score
        return highest_poi_score

    def calculateContinuousTeamFitness(self, poi_colony: Optional[POIColony], position_history: List[np.ndarray]):
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

    def calculateCounterfactualTeamFitness(self, boid_id: int, position_history: List[np.ndarray]):
        # Remove the specified boid's trajectory from the position history and calculate the fitness

        # First just copy the position_history as a np array. Np array makes it easier to slice
        counterfactual_position_history = np.array(position_history).copy()
        # Array is (timesteps, agents, xy)
        counterfactual_position_history = np.concatenate(
            (counterfactual_position_history[:,:boid_id,:],
            counterfactual_position_history[:,boid_id+1:,:]),
            axis=1
        )
        # Special case where there is only one agent and removing it causes
        # the counterfactual position history to be empty
        if counterfactual_position_history.shape[1] == 0:
            return 0.0
        else:
            return self.calculateContinuousTeamFitness(poi_colony=None, position_history=counterfactual_position_history)
    
    def calculateDifferenceEvaluations(self, position_history=List[np.ndarray]):
        G = self.calculateContinuousTeamFitness(poi_colony=None, position_history=position_history)
        difference_evaluations = []
        for leader in self.boids_colony.getLeaders():
            G_c = self.calculateCounterfactualTeamFitness(leader.id, position_history)
            D = G-G_c
            difference_evaluations.append(D)
        return difference_evaluations

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
