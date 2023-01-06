from copy import deepcopy
from typing import List, Optional

from leader_follower.agents.boids_colony import BoidsColony, Boid
from leader_follower.math_helpers import argmax
from leader_follower.agents.poi_colony import POIColony


class FitnessCalculator:
    def __init__(self, poi_colony: POIColony, boids_colony: BoidsColony, fitness):
        fitness_mapping = {
            'difference_leaders_followers': self.difference_leaders_followers,
            'difference_leaders': self.difference_leaders,
            'difference_proximity': self.difference_proximity,
        }

        self.fitness_func = fitness_mapping.get(fitness, self.difference_leaders_followers)
        self.poi_colony = poi_colony
        self.boids_colony = boids_colony
        return

    def global_discrete(self, poi_colony: Optional[POIColony] = None):
        if poi_colony is None:
            poi_colony = self.poi_colony
        return float(poi_colony.num_observed()) / float(poi_colony.num_pois)

    def global_continuous(self, poi_colony: Optional[POIColony] = None):
        if poi_colony is None:
            poi_colony = self.poi_colony
        return float(poi_colony.num_observed()) / float(poi_colony.num_pois)

    def global_stepwise_continuous(self, poi_colony: Optional[POIColony] = None):
        if poi_colony is None:
            poi_colony = self.poi_colony
        return float(poi_colony.num_observed()) / float(poi_colony.num_pois)

    def difference_leaders_followers(self, leader: Boid, assigned_follower_ids: List[int]):
        # Make a copy of the POI manager and all POIs
        poi_colony_copy = deepcopy(self.poi_colony)
        # Determine if POI would be observed without this agent and its followers
        all_removed_ids = assigned_follower_ids + [leader.id]
        for poi in poi_colony_copy.pois:
            # Set observation to False
            poi.observed = False
            # Check each group that observed this poi
            for group in poi.observation_list:
                # Recreate this group but with the leader and followers removed
                # If the leaders and followers were not in this group, then this is just
                # a copy of the original group
                difference_group = [gid for gid in group if gid not in all_removed_ids]
                # If the coupling requirement is still satisfied, then set this poi as observed
                if len(difference_group) >= self.poi_colony.coupling:
                    poi.observed = True
                    break
        return self.global_discrete() - self.global_discrete(poi_colony_copy)

    def difference_leaders(self, leader: Boid, assigned_follower_ids: List[int]):
        # Make a copy of the POI manager and all POIs
        poi_colony_copy = deepcopy(self.poi_colony)
        # Determine if POI would be observed without this agent
        all_removed_ids = [leader.id]
        for poi in poi_colony_copy.pois:
            # Set observation to False
            poi.observed = False
            # Check each group that observed this poi
            for group in poi.observation_list:
                # Recreate this group but with the leader removed
                # If the leaders, then this is just a copy of the original group
                difference_group = [gid for gid in group if gid not in all_removed_ids]
                # If the coupling requirement is still satisfied, then set this poi as observed
                if len(difference_group) >= self.poi_colony.coupling:
                    poi.observed = True
                    break
        return self.global_discrete() - self.global_discrete(poi_colony_copy)

    def difference_proximity(self, leader: Boid, assigned_follower_ids: List[int]):
        # expect to see difference when running golf and hotel
        # hotel - expect leader to not learn to slow down/shepard followers correctly
        # todo implement proximity rewards
        return

    def difference_evaluations(self):
        # Assign followers to each leader
        all_assigned_followers = [[] for _ in range(self.boids_colony.bounds.num_leaders)]
        for follower in self.boids_colony.followers():
            # Get the id of the max number in the influence list
            #   (this is the id of the leader that influenced this follower the most)
            # todo use np.argmax
            all_assigned_followers[argmax(follower.leader_influence)].append(follower.id)
        difference_rewards = []
        for leader, assigned_followers in zip(self.boids_colony.leaders(), all_assigned_followers):
            reward = self.fitness_func(leader, assigned_followers)
            # reward = self.difference_no_leaders_followers(leader, assigned_followers)
            difference_rewards.append(reward)
        return difference_rewards
