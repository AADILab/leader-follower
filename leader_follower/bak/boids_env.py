import copy
import functools
from enum import IntEnum
from typing import Dict, Optional, Union, List

import numpy as np
from gym import Env
from gym.core import RenderFrame
from gym.spaces import Box

from leader_follower.agent import Leader
from leader_follower.bak.poi import POI
from leader_follower.bak.boids_colony import Boid


def get_team_fitness(poi_colony: list[POI]):
    return float(poi_colony.numObserved()) / float(poi_colony.num_pois)


def calculate_difference_evaluation(poi_colony: list[POI], leader, assigned_follower_ids: list[int]):
    # Make a copy of the POI manager and all POIs
    poi_colony_copy = copy.deepcopy(poi_colony)
    all_removed_ids = assigned_follower_ids + [leader.id]
    for poi in poi_colony_copy.pois:
        # Determine if POI would be observed without this agent and its followers
        # Set observation to False
        poi.observed = False
        # Check each group that observed this poi
        for group in poi.observation_list:
            # Recreate this group but with the leader and followers removed
            # If the leaders and followers were not in this group, then this is just
            # a copy of the original group
            difference_group = [each_id for each_id in group if each_id not in all_removed_ids]
            # If the coupling requirement is still satisfied, then set this poi as observed
            if len(difference_group) >= poi_colony.coupling:
                poi.observed = True
                break
    return get_team_fitness(poi_colony) - get_team_fitness(poi_colony_copy)


def calculate_difference_evaluations(boids_colony: list):
    # Assign followers to each leader
    all_assigned_followers = [[] for _ in range(boids_colony.bounds.num_leaders)]
    for follower in boids_colony.getFollowers():
        # Get the id of the max number in the influence list
        #   this is the id of the leader that influenced this follower the most
        all_assigned_followers[np.argmax(follower.leader_influence)].append(follower.id)
    difference_rewards = []
    for leader, assigned_followers in zip(boids_colony.getLeaders(), all_assigned_followers):
        difference_rewards.append(calculate_difference_evaluation(leader, assigned_followers))
    return difference_rewards


class StateBounds:
    def __init__(self, x_size, y_size,
                 min_velocity, max_velocity, max_acceleration, max_angular_velocity,
                 num_leaders, num_followers, **kwargs) -> None:
        self.map_dimensions = np.array([x_size, y_size], dtype=np.float64)
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity
        self.num_leaders = num_leaders
        self.num_followers = num_followers
        return

    def num_total(self):
        return self.num_leaders + self.num_followers


class RenderMode(IntEnum):
    human = 0
    none = 1


class BoidsEnv(Env):


    metadata = {"render_modes": ["human", "none"], "name": "boids_environment"}

    def __init__(
            self, max_steps: int, leaders: list[Leader], boids: list[Boid], pois: list[POI], state_bounds, **kwargs):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self.unused_params = kwargs

        self.state_bounds = state_bounds
        self.leaders = leaders
        self.boids = boids
        self.pois = pois

        self.max_steps = max_steps
        self.num_steps = 0
        return

    def state(self) -> np.ndarray:
        pass

    def seed(self, seed=None):
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return self.observation_manager.getObservationSpace()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        # Action space is desired velocity, desired heading
        return Box(
            low=np.array([self.state_bounds.min_velocity, -np.pi], dtype=np.float64),
            high=np.array([self.state_bounds.max_velocity, np.pi], dtype=np.float64),
            dtype=np.float64
        )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        """
        np.random.seed(seed)
        self.num_steps = 0

        self.leaders = [each_leader.reset() for each_leader in self.leaders]
        self.leaders = [each_boid.reset() for each_boid in self.boids]
        self.leaders = [each_poi.reset() for each_poi in self.pois]

        observations = self.get_observations()
        return observations

    def get_observations(self):
        all_observations = self.observation_manager.getAllObservations()
        return all_observations

        # observations = {}
        # for agent, observation in zip(self.agents, all_observations):
        #     observations[agent] = observation
        # return observations

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # Format leader actions
        leader_desired_velocities = np.zeros(self.state_bounds.num_leaders)
        leader_desired_delta_headings = np.zeros(self.state_bounds.num_leaders)

        # Populate actions from dictionary
        for agent_id, agent_name in enumerate(self.agents):
            leader_desired_velocities[agent_id] = actions[agent_name][0]
            leader_desired_delta_headings[agent_id] = actions[agent_name][1]

        # Step forward simulation with leader actions
        self.boids_colony.step(
            leader_desired_velocities=leader_desired_velocities,
            leader_desired_delta_headings=leader_desired_delta_headings
        )

        # Get leader observations
        observations = self.get_observations()

        # Update POIs
        self.poi_colony.updatePois(self.boids_colony.state)

        # Step forward and check if simulation is done
        self.num_steps += 1
        env_done = self.num_steps >= self.max_steps
        # if self.num_steps >= self.max_steps:
        #     env_done = True
        # else:
        #     env_done = False

        # Update all agent dones with environment done
        dones = {agent: env_done for agent in self.agents}

        # Update infos for agents. Not sure what would go here but it
        # seems to be expected by pettingzoo
        infos = {agent: {} for agent in self.agents}

        # Calculate fitnesses
        rewards = {agent: 0.0 for agent in self.agents}
        rewards["team"] = 0.0
        if env_done:
            rewards = {
                agent: reward
                for agent, reward
                in zip(self.agents, calculate_difference_evaluations(self.leader_colony))
            }
            rewards["team"] = get_team_fitness(self.pois)

        # Return leader observations, fitnesses, whether simulation is done, and misc info
        return observations, rewards, dones, infos
