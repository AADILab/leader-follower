"""
@title

@description

"""
import abc
from abc import ABC

import numpy as np
import torch
from gym.vector.utils import spaces

from leader_follower.learn.neural_network import NeuralNetwork


class Agent(ABC):

    @property
    def state(self):
        return self.state_history[-1]

    @property
    def observation(self):
        return self.action_history[-1]

    @property
    def action(self):
        return self.observation_history[-1]

    def __init__(self, agent_id: int, brain, location: tuple, velocity: tuple,
                 sensor_resolution, observation_radius: float, value: float):
        self.name = f'agent_{agent_id}'
        self.id = agent_id
        self.type = None

        # lower/upper bounds in x, lower/upper bounds in y
        self.velocity_range = ((0, 1), (0, 1))

        self.brain = brain
        self.sensor_resolution = sensor_resolution
        self.observation_radius = observation_radius
        self.value = value

        self._initial_location = location
        self._initial_velocity = velocity

        self.location = location
        self.velocity = velocity

        # location, velocity
        state = np.asarray([location, velocity])
        self.state_history: list[np.ndarray] = [state]

        # observation history is the record of observations passed in to `get_action()`
        self.observation_history = []
        # action history is the record of actions computed by `get_action()`
        self.action_history: list[np.ndarray] = []
        return

    def __repr__(self):
        return f'{self.name=}, {self.state=}'

    def reset(self):
        self.location = self._initial_location
        self.velocity = self._initial_velocity

        state = np.asarray([self.location, self.velocity])
        self.state_history: list[np.ndarray] = [state]
        self.observation_history = []
        self.action_history: list[np.ndarray] = []
        return

    def relative(self, end_agent):
        assert isinstance(end_agent, Agent)
        start_loc = self.location
        end_loc = end_agent.location
        assert len(start_loc) == len(end_loc)

        dx = end_loc[0] - start_loc[0]
        dy = end_loc[1] - start_loc[1]
        angle = np.arctan2(dy, dx)
        angle = np.degrees(angle)
        angle = angle % 360

        dist = np.linalg.norm(np.asarray(end_loc) - np.asarray(start_loc))
        return angle, dist

    def observable_agents(self, relative_agents):
        """
        observable_agents

        :param relative_agents:
        :return:
        """
        bins = []
        for idx, agent in enumerate(relative_agents):
            assert isinstance(agent, Agent)
            if agent == self:
                continue

            angle, dist = self.relative(agent)
            if dist <= self.observation_radius:
                bins.append(agent)
        return bins

    @abc.abstractmethod
    def sense(self, relative_agents):
        return NotImplemented

    @abc.abstractmethod
    def observation_space(self):
        return NotImplemented

    @abc.abstractmethod
    def action_space(self):
        return NotImplemented

    @abc.abstractmethod
    def get_action(self, observation):
        return NotImplemented

class Leader(Agent):

    @property
    def active_policy(self):
        return self.brain[self.policy_idx]

    def __init__(self, agent_id, policy_population: list[NeuralNetwork], location, velocity, sensor_resolution, observation_radius, value):
        # agent_id: int, brain, location: tuple, velocity: tuple, sensor_resolution, observation_radius: float, value: float
        super().__init__(agent_id, policy_population, location, velocity, sensor_resolution, observation_radius, value)
        self.name = f'leader_{agent_id}'
        self.type = 'learner'

        self.policy_idx = 0

        self.n_in = 4
        self.n_out = 4
        return

    def observation_space(self):
        sensor_range = spaces.Box(low=0, high=np.inf, shape=(self.sensor_resolution,), dtype=np.float64)
        return sensor_range

    def action_space(self):
        actions = spaces.Box(
            low=np.array([self.velocity_range[0], self.velocity_range[0]], dtype=np.float64),
            high=np.array([self.velocity_range[1], self.velocity_range[1]], dtype=np.float64),
            dtype=np.float64
        )
        return actions


    def sense(self, relative_agents, sensor_resolution=None, offset=False):
        """
        Calculates which pois, leaders, and follower go into which d-hyperoctant, where d is the state
        resolution of the environment.

        :param relative_agents:
        :param sensor_resolution:
        :param offset:
        :return:
        """
        obs_agents = Agent.observable_agents(self, relative_agents)

        bin_size = 360 / self.sensor_resolution
        if offset:
            offset = 360 / (self.sensor_resolution * 2)
            bin_size = offset * 2

        octant_bins = np.zeros((2, self.sensor_resolution))
        counts = np.zeros((2, self.sensor_resolution))
        for idx, agent in enumerate(obs_agents):
            agent_type_idx = 0 if isinstance(agent, Poi) else 1
            angle, dist = self.relative(agent)
            bin_idx = int(np.floor(angle / bin_size) % self.sensor_resolution)
            octant_bins[agent_type_idx, bin_idx] += agent.value / max(dist, 0.01)
            counts[agent_type_idx, bin_idx] += 1

        # todo fix when delta_time = 0.1
        #       RuntimeWarning: invalid value encountered in divide
        #       octant_bins = np.divide(octant_bins, counts)
        octant_bins = np.divide(octant_bins, counts)
        octant_bins = np.nan_to_num(octant_bins)
        octant_bins = octant_bins.flatten()
        return octant_bins


    def set_policy(self, idx):
        self.policy_idx = idx
        return

    def add_policy(self, policy):
        self.brain.append(policy)
        return

    def remove_policy(self, idx=None):
        if not idx:
            idx = self.policy_idx
        policy = self.brain.pop(idx)
        return policy

    def get_action(self, observation):
        """
        Computes the x and y vectors using the active policy and the passed in observation.

        :param observation:
        :return:
        """
        active_policy = self.brain[self.policy_idx]
        with torch.no_grad():
            action = active_policy(observation)
            action = action.numpy()
        return action


class Follower(Agent):

    def __init__(self, agent_id, update_rule, location, velocity, sensor_resolution, observation_radius, value):
        # agent_id: int, brain, location: tuple, velocity: tuple, sensor_resolution, observation_radius: float, value: float
        super().__init__(agent_id, update_rule, location, velocity, sensor_resolution, observation_radius, value)
        self.name = f'follower_{agent_id}'
        self.type = 'actor'
        return

    def observation_space(self):
        sensor_range = spaces.Box(low=-np.inf, high=np.inf, shape=(3, 2), dtype=np.float64)
        return sensor_range

    def action_space(self):
        actions = spaces.Box(
            low=np.array([self.velocity_range[0], self.velocity_range[0]], dtype=np.float64),
            high=np.array([self.velocity_range[1], self.velocity_range[1]], dtype=np.float64),
            dtype=np.float64
        )
        return actions

    def __rule_observation(self, relative_agents, rule):
        # todo
        self.observation_radius = rule.radius
        attraction_agents = Agent.observable_agents(self, relative_agents)

        if len(attraction_agents) > 0:
            locs = [each_agent.location for each_agent in attraction_agents]
            vels = [each_agent.velocity for each_agent in attraction_agents]
        else:
            locs = [[0, 0]]
            vels = [[0, 0]]

        avg_locs = np.average(locs, axis=0)
        avg_vels = np.average(vels, axis=0)
        bins = np.asarray([avg_locs, avg_vels])
        return bins

    def sense(self, relative_agents):
        """
        agent_velocities
        Finds the average velocity and acceleration of all the agents within the observation radius of the base agent.

        :param relative_agents:
        :return:
        """
        repulsion_bins = self.__rule_observation(relative_agents, self.repulsion)
        attraction_bins = self.__rule_observation(relative_agents, self.attraction)
        alignment_bins = self.__rule_observation(relative_agents, self.alignment)
        bins = np.concatenate([repulsion_bins, attraction_bins, alignment_bins])
        return bins

    def get_action(self, observation):
        v1 = self.update_rule(observation)

        # b.velocity = b.velocity + v1 + v2 + v3
        # b.position = b.position + b.velocity
        # todo
        action = [0.0, 0.0]
        return action


class Poi(Agent):

    @property
    def observed(self):
        # todo
        return False

    def __init__(self, agent_id, location, velocity, sensor_resolution, observation_radius, value, coupling):
        # agent_id: int, brain, location: tuple, velocity: tuple, sensor_resolution, observation_radius: float, value: float
        super().__init__(agent_id, None, location, velocity, sensor_resolution, observation_radius, value)
        self.name = f'poi_{agent_id}'
        self.type = 'poi'

        self.coupling = coupling
        return

    def observation_space(self):
        sensor_range = spaces.Box(low=0, high=self.coupling, shape=(1,))
        return sensor_range

    def action_space(self):
        actions = spaces.Box(
            low=np.array([self.velocity_range[0], self.velocity_range[0]], dtype=np.float64),
            high=np.array([self.velocity_range[1], self.velocity_range[1]], dtype=np.float64),
            dtype=np.float64
        )
        return actions

    def sense(self, relative_agents):
        obs = self.observable_agents(relative_agents)
        return obs

    def get_action(self, observation):
        action = np.array([0, 0])
        return action
