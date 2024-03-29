import functools
from typing import Dict, Optional, Union
from enum import IntEnum

from gym.spaces import Box
from pettingzoo import ParallelEnv
import numpy as np

from leaderfollower.colony_helpers import StateBounds
from leaderfollower.boid_spawner import BoidSpawner
from leaderfollower.boids_colony import BoidsColony
from leaderfollower.poi_spawner import POISpawner
from leaderfollower.poi_colony import POIColony
from leaderfollower.fitness_calculator import FitnessCalculator
from leaderfollower.observations_manager import ObservationManager
from leaderfollower.renderer import Renderer


class RenderMode(IntEnum):
    human = 0
    none = 1


class BoidsEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "none"], "name": "boids_environment"}

    def __init__(self, max_steps: int, render_mode: Union[RenderMode, str], init_seed: int, config: Dict) -> None:
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
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
            init_state=self.boid_spawner.getSpawnState(),
            bounds=self.state_bounds,
            **config["BoidsColony"]
        )
        self.poi_spawner = POISpawner(
            map_dimensions=self.map_dimensions,
            **config["POISpawner"]
        )
        self.poi_colony = POIColony(
            positions=self.poi_spawner.getSpawnPositions(),
            **config["POIColony"]
        )
        self.fitness_calculator = FitnessCalculator(
            boids_colony=self.boids_colony,
            poi_colony=self.poi_colony,
            **config["FitnessCalculator"]
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

        # Position history will be a list of the positions of all agents (leaders and followers) in the environment
        # Each element captures the positions of all agents at a particular timestep, starting with t=0
        # Each element is a np array of positions 
        self.position_history = [self.boids_colony.state.positions.copy()]

        # An array of potential value histories
        self.potential_values = []
        self.subtracted_potentials = []

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

        self.renderer.renderFrame()

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        '''
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        '''
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.num_steps = 0
        self.boids_colony.reset(reset_state=self.boid_spawner.getSpawnState())
        self.poi_colony.reset(positions=self.poi_spawner.getSpawnPositions())
        observations = self.getObservations()
        self.position_history = [self.boids_colony.state.positions.copy()]
        return observations

    def getObservations(self):
        all_observations = self.observation_manager.getAllObservations()

        observations = {}
        for agent, observation in zip(self.agents, all_observations):
            observations[agent] = observation
        return observations

    def step(self, actions):
        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''
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

        # Update the position history
        self.position_history.append(self.boids_colony.state.positions.copy())

        # Get leader observations
        observations = self.getObservations()

        # Update POIs
        self.fitness_calculator.updatePOIs()
        # self.poi_colony.updatePois(self.boids_colony.state, self.fitness_calculator)

        # Step forward and check if simulation is done
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            env_done = True
        else:
            env_done = False

        # Update all agent dones with environment done
        dones = {agent: env_done for agent in self.agents}

        # Update infos for agents. Not sure what would go here but it
        # seems to be expected by pettingzoo
        infos = {agent: {} for agent in self.agents}

        # Calculate fitnesses
        if env_done:
            G = self.fitness_calculator.calculateG(self.position_history)
            Ds = self.fitness_calculator.calculateDs(G=G, position_history=self.position_history)
            Fs = self.fitness_calculator.calculateFs(self.position_history, self.potential_values)
            self.subtracted_potentials.append(Fs[1])

            total_F = np.sum(np.array(self.subtracted_potentials), axis=0)
            #print("Total F " + str(total_F))
            Ds = (np.array(Ds) + total_F).tolist()
            #print("total D " + str(Ds))

            
            rewards = {
                agent: reward
                for agent, reward
                in zip(self.agents, Ds)
            }
            
            rewards["team"] = G

        else:
            Fs = self.fitness_calculator.calculateFs(self.position_history, self.potential_values)

            #print("potential array: " + str(self.potential_values))
            self.subtracted_potentials.append(Fs[1])
            #print("subtracted potentials: " + str(self.subtracted_potentials))
            rewards = {agent: 0.0 for agent in self.agents}
            rewards["team"] = 0.0

        # Return leader observations, fitnesses, whether simulation is done, and misc info
        return observations, rewards, dones, infos
