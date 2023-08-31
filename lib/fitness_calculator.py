from typing import List, Optional, Union
from copy import deepcopy
from enum import IntEnum

import numpy as np

from lib.poi_colony import POIColony, POI
from lib.boids_colony import BoidsColony, Boid
from lib.math_helpers import argmax, calculateDistance
from lib.np_helpers import invertInds

class FollowerSwitch(IntEnum):
    # Only use follower positions when calculating G
    UseFollowersOnly = 0
    # Use both leader and follower positions when calculating G
    UseLeadersAndFollowers = 1

class WhichG(IntEnum):
    # Inverse distance to POIs w. coupling through an episode
    Continuous = 0
    # MinContinous is continuous, but we only use the min inverse distance(s) at
    # a particular timestep. Found by iterating through each timestep in an episode
    # Note that I cap each POI score at 1 so agents cant get near inifinte score for being right on top of a POI
    MinContinuous = 1
    # MinDiscrete is the all or nothing reward structure. Basically, for each POI,
    # check if enough agents were within its observation radius at each point during the episode
    # If at any point, the coupling requirement is met, then we count that POI as observed
    # Num poi observed / total num poi
    MinDiscrete = 2
    # ContinuosObsRad is what D++ used. (Actually it's not clear if G was just for the final state-action or aggregated throughout the episode)
    # Inverse distance to POIs w. coupling through an episode. Aggregates score at each time step
    # Only count observations if they are within the observation radius of the poi
    ContinuousObsRad = 3
    # ContinuousObsRadLastStep is maybe what D++ used?
    # Inverse distance to POIs w. coupling, only uses the last step in the trajectory.
    # Only count observations if they are within the observation radius of the poi
    ContinuousObsRadLastStep = 4


class WhichD(IntEnum):
    # Don't calculate a difference reward. Just give each agent G
    G = 0
    # Use standard D where we just remove this agent's trajectory
    D = 1
    # Calculate D of this (leader) agent and its followers
    DFollow = 2
    # Literally just give the policy a fitness of zero no matter what.
    # When we compare reward signals against this one, we'll be able to know if a reward signal is "misleading"
    # A reward signal is probably misleading if a consistent reward of zero results in a better policy
    Zero = 3

class WhichF(IntEnum):
    # Don't use a potential function. Just give 0
    G = 0
    # Use a potential function for leaders grabbing followers
    FCouple = 1

    FCPoi = 2

    FDistance = 3

    FCoupleD = 4

class PotentialType(IntEnum):
    #use global potential values
    Global = 0
    #use agent specific potential values
    Agent = 1

class UseDrip(IntEnum):
    Drip = 0
    Regular = 1

class FitnessCalculator():
    def __init__(self, poi_colony: POIColony, boids_colony: BoidsColony, which_G: Union[WhichG, str], which_D: Union[WhichD, str], which_F: Union[WhichF, str] = WhichF.G, potential_type: Union[PotentialType, str] = PotentialType.Global, use_drip: Union[UseDrip, str] = UseDrip.Regular, follower_switch: Union[FollowerSwitch, str] = FollowerSwitch.UseLeadersAndFollowers) -> None:
        self.poi_colony = poi_colony
        self.boids_colony = boids_colony
        
        if type(which_G) == str:
            which_G = WhichG[which_G]
        if type(which_D) == str:
            which_D = WhichD[which_D]
        if type(follower_switch) == str:
            follower_switch = FollowerSwitch[follower_switch]
        if type(which_F) == str:
            which_F = WhichF[which_F]
        if type(potential_type) == str:
            potential_type = PotentialType[potential_type]
        if type(use_drip) == str:
            use_drip = UseDrip[use_drip]
            # if(use_drip == UseDrip.Split):
            #     #Have to make potential type Global for DRiP
            #     potential_type = PotentialType["Global"]
        
        self.which_G = which_G
        self.which_D = which_D
        self.which_F = which_F
        # follower_switch has a default setting for backwards compatability
        self.follower_switch = follower_switch
        self.potential_type = potential_type
        self.use_drip = use_drip

    def calculateG(self, position_history: List[np.ndarray]):
        # Remove leader trajectories if we are only evaluating based on followers
        if self.follower_switch == FollowerSwitch.UseFollowersOnly:
            # Note: follower trajectories are on the right side of the position history
            # I am now converting position_history to an array here so hopefully this doesn't cause any problems
            position_history = np.array(position_history)[:, self.boids_colony.bounds.num_leaders:]

        if self.which_G == WhichG.Continuous:
            raise NotImplementedError()
        elif self.which_G == WhichG.MinContinuous:
            return self.calculateMinContinuousG(position_history)
        elif self.which_G == WhichG.MinDiscrete:
            return self.calculateMinDiscreteG(position_history)
        elif self.which_G == WhichG.ContinuousObsRad:
            return self.calculateContinuousObsRadG(position_history)
        elif self.which_G == WhichG.ContinuousObsRadLastStep:
            return self.calculateContinuousObsRadLastStepG(position_history)
        
    def calculateContinuousObsRadLastStepPOIScore(self, poi: POI, position_history: List[np.ndarray]):
        # Grab the last positions of agents in the environment
        agent_positions = position_history[-1]
        distances = self.calculateDistances(poi, agent_positions)
        distances_sorted = np.sort(distances)
        distances_sorted[distances_sorted < 1] = 1
        if len(distances_sorted) >= self.poi_colony.coupling \
            and np.all(distances_sorted[:self.poi_colony.coupling]<=self.poi_colony.observation_radius):
            poi_score = 1/( (1/self.poi_colony.coupling) * (np.sum(distances_sorted[:self.poi_colony.coupling])) )
        else:
            poi_score = 0
        return poi_score
    
    def calculateContinuousObsRadLastStepG(self, position_history: List[np.ndarray]):
        total_score = 0
        for poi in self.poi_colony.pois:
            last_step_poi_score = self.calculateContinuousObsRadLastStepPOIScore(poi, position_history)
            total_score += last_step_poi_score
        # Normalize score by number of pois
        return total_score / self.poi_colony.num_pois
    
    def calculateContinuousObsRadG(self, position_history: List[np.ndarray]):
        total_score = 0
        for poi in self.poi_colony.pois:
            rolling_poi_score = self.calculateContinuousObsRadPOIScore(poi, position_history)
            total_score += rolling_poi_score
        # Normalize score by number of pois
        return total_score / self.poi_colony.num_pois

    def calculateContinuousObsRadPOIScore(self, poi: POI, position_history: List[np.ndarray]):
        # Go through position history. Calculate score for each timestep. Aggregate scores
        # Coupling determines how many agents we look at to determine the observation value
        # The score is 0 if not enough agents are within the observation radius

        rolling_poi_score = 0
        for t, agent_positions in enumerate(position_history):
            distances = self.calculateDistances(poi, agent_positions)
            # Coupling determines how many distances we use here
            distances_sorted = np.sort(distances)
            distances_sorted[distances_sorted < 1] = 1
            # Make sure to only calculate a score if the coupling requirement is met
            # AND if all the distances were within the observation radius of that poi
            if len(distances_sorted) >= self.poi_colony.coupling \
                and np.all(distances_sorted[:self.poi_colony.coupling]<=self.poi_colony.observation_radius):
                poi_score = 1/( (1/self.poi_colony.coupling) * (np.sum(distances_sorted[:self.poi_colony.coupling])) )
            else:
                poi_score = 0
            rolling_poi_score += poi_score
        # Normalize by the number of timesteps
        return rolling_poi_score / len(position_history)
    
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

        elif self.which_D == WhichD.Zero:
            # Literally just return zeros. Floating point in case it matters
            return [0. for leader in self.boids_colony.getLeaders()]
            
    def calculateD(self, G: float, ids_to_remove: int, position_history: List[np.ndarray]):
        G_c = self.calculateCounterfactualG(position_history, ids_to_remove)
        if G_c > G:
            raise Exception("G_c > G | Counterfactual G was greater than actual G. Something went wrong calculating the counterfactual")
        return G-G_c
    
    def calculateDistances(self, poi, agent_positions):
        # Calculate the distances for one timestep for a given poi and positions of all agents
        return np.linalg.norm(poi.position - agent_positions, axis=1)

    def calculateMinContPOIScore(self, poi: POI, position_history: List[np.ndarray]):
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
            highest_poi_score = self.calculateMinContPOIScore(poi, position_history)
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
        # If we're only counting followers, then it means that when we calculate G,
        # the leader trajectories will automatically be removed unless we specifically
        # tell the fitness calculator to not do that
        if self.follower_switch == FollowerSwitch.UseFollowersOnly:
            # print("follower switch>")
            # First just copy the position_history as a np array. Np array makes it easier to slice
            counterfactual_position_history = np.array(position_history).copy()
            # Add the leaders to what we need to remove
            ids_to_remove += list(range(self.boids_colony.bounds.num_leaders))
            # print("Ids to remove: ", ids_to_remove)
            # Invert ids to remove to which ones we're keeping
            ids_to_keep = invertInds(counterfactual_position_history.shape[1], ids_to_remove)
            # Turn the follower switch off. This is so G does not try to remove the leaders again
            self.follower_switch = FollowerSwitch.UseLeadersAndFollowers
            # Calculate G but with those ids removed. Hence, G_c
            G_c = self.calculateG(counterfactual_position_history[:,ids_to_keep,:])
            # Turn follower switch back on
            self.follower_switch = FollowerSwitch.UseFollowersOnly
            # Return counterfactual G
            return G_c
        else:
            # First just copy the position_history as a np array. Np array makes it easier to slice
            counterfactual_position_history = np.array(position_history).copy()
            # Invert ids to remove to which ones we're keeping
            ids_to_keep = invertInds(counterfactual_position_history.shape[1], ids_to_remove)
            # Calculate G but with those ids removed. Hence, G_c
            return self.calculateG(counterfactual_position_history[:,ids_to_keep,:])
    
    def calculateFDifferences(self, position_history, potential_values):
        if self.which_F == WhichF.G:
            #should never run this because DRiP means this won't be used, but just in case
            current_potentials = 0 #if self.potential_type == PotentialType.Global else [0 for i in range(self.boids_colony.bounds.num_leaders)]

            return current_potentials

        potential_differences = []

        #need to get the global F value to subtract by to get agent-specific
        if(self.which_F == WhichF.FCouple):
            global_f = self.calculate_coupling_potential()
        elif(self.which_F == WhichF.FDistance):
            global_f = self.calculate_distance_potential(position_history)
        elif(self.which_F == WhichF.FCoupleD):
            global_f = self.calculate_coupled_distance_potential(position_history)

        for leader in self.boids_colony.getLeaders():
            #replacing the influence list with no followers for this leader
            new_influence_list = self.boids_colony.num_followers_influenced[:]
            new_influence_list[leader.id] = set()

            if(self.which_F == WhichF.FCouple):
                counterfactual_f = self.calculate_coupling_potential(new_influence_list)
            elif(self.which_F == WhichF.FDistance):
                counterfactual_f = self.calculate_distance_potential(position_history, new_influence_list)
            elif(self.which_F == WhichF.FCoupleD):
                counterfactual_f = self.calculate_coupled_distance_potential(position_history, new_influence_list)
            
            potential_differences.append(global_f - counterfactual_f)
        
        return potential_differences
    
    def calculateFs(self, position_history, potential_values):
        
        if(self.use_drip == UseDrip.Drip):
            potential_values.append(self.calculateFDifferences(position_history, potential_values))
            if(len(potential_values) > 1):
                return potential_values[-1], (np.array(potential_values[-1]) - np.array(potential_values[-2])).tolist()
            else:
                return potential_values[-1], potential_values[-1]

        if self.which_F == WhichF.G:
            #Just return 0 for each leader in the case of no PBRS
            #return [0] if self.potential_type == PotentialType.Global else [0 for i in range(self.boids_colony.bounds.num_leaders)]
            current_potentials = 0 if self.potential_type == PotentialType.Global else [0 for i in range(self.boids_colony.bounds.num_leaders)]
            potential_values.append(current_potentials)

            return current_potentials, current_potentials
    
        elif self.which_F == WhichF.FCouple:
            
            potential_values.append(self.calculate_coupling_potential())

            if(self.potential_type == PotentialType.Global):
                if(len(potential_values) == 1):
                    return potential_values[-1], potential_values[-1]
                else:
                    return potential_values[-1], potential_values[-1] - potential_values[-2]
            else:
                if(len(potential_values) == 1):
                    return potential_values[-1], potential_values[-1]
                else:
                    return potential_values[-1], (np.array(potential_values[-1]) - np.array(potential_values[-2])).tolist()
        elif self.which_F == WhichF.FDistance:
            potential_values.append(self.calculate_distance_potential(position_history))

            if(self.potential_type == PotentialType.Global):
                if(len(potential_values) == 1):
                    return potential_values[-1], potential_values[-1]
                else:
                    return potential_values[-1], potential_values[-1] - potential_values[-2]
            else:
                if(len(potential_values) == 1):
                    return potential_values[-1], potential_values[-1]
                else:
                    return potential_values[-1], (np.array(potential_values[-1]) - np.array(potential_values[-2])).tolist()
        elif self.which_F == WhichF.FCoupleD:
            potential_values.append(self.calculate_coupled_distance_potential(position_history))

            if(self.potential_type == PotentialType.Global):
                if(len(potential_values) == 1):
                    return potential_values[-1], potential_values[-1]
                else:
                    return potential_values[-1], potential_values[-1] - potential_values[-2]
            else:
                if(len(potential_values) == 1):
                    return potential_values[-1], potential_values[-1]
                else:
                    return potential_values[-1], (np.array(potential_values[-1]) - np.array(potential_values[-2])).tolist()
        
    def calculate_coupling_potential(self, influence_list = None):
        # potential_values.append(num_followers_influenced)

        if(influence_list is None):
            num_followers_influenced = self.boids_colony.num_followers_influenced
        else:
            num_followers_influenced = influence_list
        
        
        current_potentials = []
        #going through to not give more reward than the max coupling 
        for i, val in enumerate(num_followers_influenced):
            if(len(val) > self.poi_colony.coupling):
                current_potentials.append(self.poi_colony.coupling)
            else:
                current_potentials.append(len(val))
        
        if(self.potential_type == PotentialType.Agent):
            #keeps potentials agent specific
            return current_potentials
        else:
            #takes the sum of coupling to become agent specific
            return sum(current_potentials)
    
    def calculate_poi_potential(self, position_history, potential_values):
        position_history = np.array(position_history)[:, :self.boids_colony.bounds.num_leaders]
        agent_positions = position_history[-1]
        for poi in self.poi_colony.pois: 
            distances = self.calculateDistances(poi, agent_positions)
            distances_sorted = np.sort(distances)
            #distances_sorted[distances_sorted < 1] = 1

            #rewarding a leader for getting to the POI (only reward the one that is closest to this POI)
            if(distances_sorted[0] <= self.poi_colony.observation_radius):
                potential_values[-1][np.argmin(distances)] += 0.1

            # if(np.all(distances_sorted[:self.poi_colony.coupling]<=self.poi_colony.observation_radius)):
            #     potential_values[-1].append(1)
            #     return
            
    def calculate_coupled_distance_potential(self, position_history, influence_list = None):
        if(influence_list is not None):
            couple_val = self.calculate_coupling_potential(influence_list)
            distance_val = self.calculate_distance_potential(position_history, influence_list)
        else:
            couple_val = self.calculate_coupling_potential()
            distance_val = self.calculate_distance_potential(position_history)

        if(self.potential_type == PotentialType.Global):
            return couple_val + distance_val
        else:
            return (np.array(couple_val) + np.array(distance_val)).tolist()
    
    def calculate_distance_potential(self, position_history, influence_list = None):
        if(influence_list is None):
            num_followers_influenced = self.boids_colony.num_followers_influenced
        else:
            num_followers_influenced = influence_list

        current_potentials = [0 for i in range(self.boids_colony.bounds.num_leaders)]

        position_history = np.array(position_history)[:, self.boids_colony.bounds.num_leaders:]
        agent_positions = position_history[-1]
        for poi in self.poi_colony.pois: 
            distances = self.calculateDistances(poi, agent_positions)
            #distances_sorted = np.sort(distances)
            distances_sorted = np.argsort(distances)

            for index in distances_sorted:
                if(distances[index] > self.poi_colony.observation_radius):
                    break #move to next POI

                for m, k in enumerate(num_followers_influenced):
                    #indices are off by num_leaders because of positon_history on line 357

                    #only give the distance potential if the leader is near the follower
                    if((index + self.boids_colony.bounds.num_leaders) in k):
                        #print((1 / self.poi_colony.coupling))
                        current_potentials[m] += (1 / (self.poi_colony.coupling * self.poi_colony.num_pois))
        
        if(self.potential_type == PotentialType.Agent):
            #keeps potentials agent specific
            return current_potentials
        else:
            #takes the sum of coupling to become agent specific
            return sum(current_potentials)
    
    def updatePOIs(self):
        """Update POIs as observed or not for rendering purposes"""
        for poi in self.poi_colony.pois:
            if self.follower_switch == FollowerSwitch.UseLeadersAndFollowers:
                distances = calculateDistance(poi.position, self.boids_colony.state.positions)
            else:
                distances = calculateDistance(poi.position, self.boids_colony.state.positions[self.boids_colony.bounds.num_leaders:])
            num_observations = np.sum(distances<=self.poi_colony.observation_radius)
            if num_observations >= self.poi_colony.coupling:
                poi.observed = True
                # Legacy code for back when I was using the all or nothing G
                # # Get ids of swarm members that observed this poi
                # observer_ids = np.nonzero(distances<=self.observation_radius)[0]
                # poi.observation_list.append(observer_ids)
            # Turn the poi back to unobserved if we only care about the last step
            elif self.which_G == WhichG.ContinuousObsRadLastStep:
                    poi.observed = False

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
