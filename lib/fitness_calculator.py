from typing import List, Optional
from copy import deepcopy

from lib.poi_colony import POIColony, POI
from lib.boids_colony import BoidsColony, Boid
from lib.math_helpers import argmax

class FitnessCalculator():
    def __init__(self, poi_colony: POIColony, boids_colony: BoidsColony) -> None:
        self.poi_colony = poi_colony
        self.boids_colony = boids_colony

    def getTeamFitness(self, poi_colony: Optional[POIColony] = None):
        if poi_colony is None:
            poi_colony = self.poi_colony
        return float(poi_colony.numObserved())/float(poi_colony.num_pois)

    def calculateDifferenceEvaluation(self, leader: Boid, assigned_follower_ids: List[int]):
        # Make a copy of the POI manager and all POIs
        poi_colony_copy = deepcopy(self.poi_colony)
        all_removed_ids = assigned_follower_ids+[leader.id]
        for poi in poi_colony_copy.pois:
            # Determine if POI would be observed without this agent and its followers
            # Set observation to False
            poi.observed = False
            # Check each group that observed this poi
            for group in poi.observation_list:
                # Recreate this group but with the leader and followers removed
                # If the leaders and followers were not in this group, then this is just
                # a copy of the original group
                difference_group = [id for id in group if id not in all_removed_ids]
                # If the coupling requirement is still satisfied, then set this poi as observed
                if len(difference_group) >= self.poi_colony.coupling:
                    poi.observed = True
                    break
        return self.getTeamFitness() - self.getTeamFitness(poi_colony_copy)

    def calculateDifferenceEvaluations(self):
        # Assign followers to each leader
        all_assigned_followers = [[] for _ in range(self.boids_colony.bounds.num_leaders)]
        for follower in self.boids_colony.getFollowers():
            # Get the id of the max number in the influence list (this is the id of the leader that influenced this follower the most)
            all_assigned_followers[argmax(follower.leader_influence)].append(follower.id)
        difference_rewards = []
        for leader, assigned_followers in zip(self.boids_colony.getLeaders(), all_assigned_followers):
            difference_rewards.append(self.calculateDifferenceEvaluation(leader, assigned_followers))
        return difference_rewards
