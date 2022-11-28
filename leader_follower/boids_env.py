import functools
from typing import Dict, Optional, Union
from enum import IntEnum

from gym.spaces import Box
from pettingzoo import ParallelEnv
import numpy as np

from leader_follower.colony_helpers import StateBounds
from leader_follower.boid_spawner import BoidSpawner
from leader_follower.boids_colony import BoidsColony
from leader_follower.poi_spawner import POISpawner
from leader_follower.poi_colony import POIColony
from leader_follower.fitness_calculator import FitnessCalculator
from leader_follower.observations_manager import ObservationManager
from leader_follower.renderer import Renderer


class RenderMode(IntEnum):
    human = 0
    none = 1


class BoidsEnv(ParallelEnv):
    def seed(self, seed=None):
        pass

    def state(self) -> np.ndarray:
        pass

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    metadata = {"render_modes": ["human", "none"], "name": "boids_environment"}

    def __init__(self, max_steps: int, render_mode: Union[RenderMode, str], config: Dict, **kwargs) -> None:
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        # todo yell at Ever
        # np.random.seed(init_seed)
        self.num_steps = 0
        self.max_steps = max_steps
        if type(render_mode) == str:
            render_mode = RenderMode[render_mode]
        self.render_mode = render_mode

        self.map_dimensions = np.array([config["map_dimensions"]["x"], config["map_dimensions"]["y"]], dtype=np.float64)
        self.state_bounds = StateBounds(
            map_dimensions=self.map_dimensions,
            **config["StateBounds"]
        )
        self.boid_spawner = BoidSpawner(
            bounds=self.state_bounds,
            **config["BoidSpawner"]
        )
        self.boids_colony = BoidsColony(
            init_state=self.boid_spawner.spawn_state(),
            bounds=self.state_bounds,
            **config["BoidsColony"]
        )
        self.poi_spawner = POISpawner(
            map_dimensions=self.map_dimensions,
            **config["POISpawner"]
        )
        self.poi_colony = POIColony(
            positions=self.poi_spawner.get_spawn_positions(),
            **config["POIColony"]
        )
        self.fitness_calculator = FitnessCalculator(
            boids_colony=self.boids_colony,
            poi_colony=self.poi_colony,
            fitness=FitnessCalculator.difference_no_leaders_followers
        )
        self.observation_manager = ObservationManager(
            boids_colony=self.boids_colony,
            poi_colony=self.poi_colony,
            map_dimensions=self.map_dimensions,
            **config["ObservationManager"]
        )
        if render_mode.value == RenderMode.human:
            self.renderer = Renderer(
                boids_colony=self.boids_colony,
                poi_colony=self.poi_colony,
                observation_manager=self.observation_manager,
                **config["Renderer"]
            )
        else:
            self.renderer = None
        self.possible_agents = ["leader_" + str(n) for n in range(self.state_bounds.num_leaders)]
        self.agents = self.possible_agents
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.config = config
        return

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return self.observation_manager.get_observation_space()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent=None):
        # Action space is desired velocity, desired heading
        return Box(
            low=np.array([self.state_bounds.min_velocity, -np.pi], dtype=np.float64),
            high=np.array([self.state_bounds.max_velocity, np.pi], dtype=np.float64),
            dtype=np.float64
        )

    def render(self, mode: Optional[Union[RenderMode, str]] = 'human'):
        if type(mode) == str:
            mode = RenderMode[mode]

        if mode.value != RenderMode.none and self.render_mode.value == RenderMode.none:
            self.renderer = Renderer(
                boids_colony=self.boids_colony,
                poi_colony=self.poi_colony,
                observation_manager=self.observation_manager,
                **self.config["Renderer"]
            )
            self.render_mode = mode

        self.renderer.render_frame()
        return

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        :param seed:
        :param options:
        :param kwargs:
        """
        if seed is not None:
            # todo yell at every about possible issues with how this flows
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.num_steps = 0
        self.boids_colony.reset(reset_state=self.boid_spawner.spawn_state())
        self.poi_colony.reset(positions=self.poi_spawner.get_spawn_positions())
        observations = self.observations()
        return observations

    def observations(self):
        all_observations = self.observation_manager.get_all_observations()

        observations = {}
        for agent, observation in zip(self.agents, all_observations):
            observations[agent] = observation
        return observations

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
        observations = self.observations()

        # Update POIs
        self.poi_colony.update(self.boids_colony.state)

        # Step forward and check if simulation is done
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            env_done = True
        else:
            env_done = False

        # Update all agent dones with environment done
        dones = {agent: env_done for agent in self.agents}

        # Update infos for agents. Not sure what would go here, but it seems to be expected by pettingzoo
        infos = {agent: {} for agent in self.agents}

        # Calculate fitnesses
        if env_done:
            rewards = {
                agent: reward
                for agent, reward
                in zip(self.agents, self.fitness_calculator.difference_evaluations())
            }
            rewards["team"] = self.fitness_calculator.global_discrete()
        else:
            rewards = {agent: 0.0 for agent in self.agents}
            rewards["team"] = 0.0

        # Return leader observations, fitnesses, whether simulation is done, and misc info
        return observations, rewards, dones, infos
