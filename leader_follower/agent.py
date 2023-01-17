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

    def __init__(self, agent_id: int, location: tuple, velocity: tuple, sensor_resolution: int, value: float):
        self.name = f'agent_{agent_id}'
        self.id = agent_id
        self.type = None

        # lower/upper bounds agent is able to move
        # same for both x and y directions
        self.velocity_range = np.array((0, 1))
        self.sensor_resolution = sensor_resolution
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
        # add very small amount of gaussian noise to make the locations unequal
        assert len(start_loc) == len(end_loc)
        # assert start_loc != end_loc

        dx = end_loc[0] - start_loc[0]
        dy = end_loc[1] - start_loc[1]
        angle = np.arctan2(dy, dx)
        angle = np.degrees(angle)
        angle = angle % 360

        dist = np.linalg.norm(np.asarray(end_loc) - np.asarray(start_loc))
        return angle, dist

    def observable_agents(self, relative_agents, observation_radius):
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
            if dist <= observation_radius:
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

    def __init__(self, agent_id, location, velocity, sensor_resolution, value, observation_radius,
                 policy: NeuralNetwork | None):
        # agent_id: int, location: tuple, velocity: tuple, sensor_resolution, observation_radius: float, value: float
        super().__init__(agent_id, location, velocity, sensor_resolution, value)
        self.name = f'leader_{agent_id}'
        self.type = 'learner'

        self.observation_radius = observation_radius
        self.policy = policy

        self.n_in = self.sensor_resolution * 2
        self.n_out = 2
        return

    def observation_space(self):
        sensor_range = spaces.Box(
            low=0, high=np.inf,
            shape=(self.sensor_resolution, 2), dtype=np.float64
        )
        return sensor_range

    def action_space(self):
        action_range = spaces.Box(
            low=self.velocity_range[0], high=self.velocity_range[1],
            shape=(2,), dtype=np.float64
        )
        return action_range

    def sense(self, other_agents, sensor_resolution=None, offset=False):
        """
        Calculates which pois, leaders, and follower go into which d-hyperoctant, where d is the state
        resolution of the environment.

        :param other_agents:
        :param sensor_resolution:
        :param offset:
        :return:
        """
        obs_agents = Agent.observable_agents(self, other_agents, self.observation_radius)

        bin_size = 360 / self.sensor_resolution
        if offset:
            offset = 360 / (self.sensor_resolution * 2)
            bin_size = offset * 2

        octant_bins = np.zeros((2, self.sensor_resolution))
        counts = np.ones((2, self.sensor_resolution))
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

    def get_action(self, observation):
        """
        Computes the x and y vectors using the active policy and the passed in observation.

        :param observation:
        :return:
        """
        active_policy = self.policy
        with torch.no_grad():
            action = active_policy(observation)
            action = action.numpy()
        return action

class Follower(Agent):

    def __init__(self, agent_id, location, velocity, sensor_resolution, value,
                 repulsion_radius, repulsion_strength, attraction_radius, attraction_strength):
        # agent_id: int, location: tuple, velocity: tuple, sensor_resolution, observation_radius: float, value: float
        super().__init__(agent_id, location, velocity, sensor_resolution, value)
        self.name = f'follower_{agent_id}'
        self.type = 'actor'

        self.repulsion_radius = repulsion_radius
        self.repulsion_strength = repulsion_strength

        self.attraction_radius = attraction_radius
        self.attraction_strength = attraction_strength

        self.__obs_rule = self.__rule_mass_center

        self.influence_history = {'repulsion': [], 'attraction': []}
        return

    def observation_space(self):
        # sum of vectors of agents in each radius
        #   repulsion
        #   attraction
        sensor_range = spaces.Box(low=-np.inf, high=np.inf, shape=(2, 2), dtype=np.float64)
        return sensor_range

    def action_space(self):
        action_range = spaces.Box(
            low=self.velocity_range[0], high=self.velocity_range[1],
            shape=(2,), dtype=np.float64
        )
        return action_range

    def __rule_loc_velocity(self, relative_agents, rule_radius):
        # todo test for correctness
        self.observation_radius = rule_radius
        rel_agents = Agent.observable_agents(self, relative_agents, rule_radius)
        rel_agents.append(self)

        locs = [each_agent.location for each_agent in rel_agents]
        vels = [each_agent.velocity for each_agent in rel_agents]

        avg_locs = np.average(locs, axis=0)
        avg_vels = np.average(vels, axis=0)
        bins = np.asarray([avg_locs, avg_vels])
        return bins

    def __rule_mass_center(self, relative_agents, rule_radius):
        self.observation_radius = rule_radius
        rel_agents = Agent.observable_agents(self, relative_agents, rule_radius)
        # adding self partially guards against when no other agents are nearby
        rel_agents.append(self)

        locs = [each_agent.location for each_agent in rel_agents]
        avg_locs = np.average(locs, axis=0)
        rel_agents.remove(self)
        return avg_locs, rel_agents

    def sense(self, relative_agents):
        """
        agent_velocities
        Finds the average velocity and acceleration of all the agents within the observation radius of the base agent.

        :param relative_agents:
        :return:
        """
        repulsion_bins, repulsion_agents = self.__obs_rule(relative_agents, self.repulsion_radius)
        attraction_bins, attraction_agents = self.__obs_rule(relative_agents, self.attraction_radius)

        self.influence_history['repulsion'].extend(repulsion_agents)
        self.influence_history['attraction'].extend(attraction_agents)

        bins = np.vstack([repulsion_bins, attraction_bins])
        return bins

    def influence_counts(self):
        repulsion_names = [agent.name for agent in self.influence_history['repulsion']]
        repulsion_counts = np.unique(repulsion_names, return_counts=True)

        attraction_names = [agent.name for agent in self.influence_history['attraction']]
        attraction_counts = np.unique(attraction_names, return_counts=True)

        repulsion_names.extend(attraction_names)
        total_counts = np.unique(repulsion_names, return_counts=True)
        return total_counts, repulsion_counts, attraction_counts

    def get_action(self, observation):
        # todo check repulsion is moving the agent in the correct direction
        repulsion_diff = np.subtract(observation[0], self.location)
        # todo bug fix
        #   RuntimeWarning: invalid value encountered in divide
        #   unit_repulsion = repulsion_diff / (repulsion_diff**2).sum()**0.5
        unit_repulsion = repulsion_diff / (repulsion_diff**2).sum()**0.5
        unit_repulsion = np.nan_to_num(unit_repulsion)
        weighted_repulsion = - unit_repulsion * self.repulsion_strength

        attraction_diff = np.subtract(observation[1], self.location)
        unit_attraction = attraction_diff / (attraction_diff ** 2).sum() ** 0.5
        unit_attraction = np.nan_to_num(unit_attraction)
        weighted_attraction = unit_attraction * self.attraction_strength

        action = weighted_attraction - weighted_repulsion
        return action

class Poi(Agent):

    def __init__(self, agent_id, location, velocity, sensor_resolution, value, observation_radius, coupling):
        # agent_id: int, location: tuple, velocity: tuple, sensor_resolution, observation_radius: float, value: float
        super().__init__(agent_id, location, velocity, sensor_resolution, value)
        self.name = f'poi_{agent_id}'
        self.type = 'poi'

        self.observation_radius = observation_radius
        self.coupling = coupling
        # flag instead of parsing history for reduce call overhead
        self.observed = False
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
        obs = self.observable_agents(relative_agents, self.observation_radius)
        # set observed flag if enough agents are observable to meet coupling requirement
        self.observed = len(obs) >= self.coupling
        return obs

    def get_action(self, observation):
        self.observation_history.append(observation)
        action = np.array([0, 0])
        return action
