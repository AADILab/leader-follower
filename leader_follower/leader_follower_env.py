import functools

import numpy as np
from pettingzoo import ParallelEnv

from leader_follower.agent import Poi, Leader, Follower


class LeaderFollowerEnv(ParallelEnv):
    metadata = {'render_modes': ['human', 'rgb_array', 'none'], 'name': 'leader_follower_environment'}

    @property
    def agent_mapping(self):
        return self._leaders | self._followers | self._pois

    @property
    @functools.lru_cache(maxsize=None)
    def observation_spaces(self):
        return {name: agent.observation_space() for name, agent in self.agent_mapping.items()}

    @property
    @functools.lru_cache(maxsize=None)
    def action_spaces(self):
        return {name: agent.action_space() for name, agent in self.agent_mapping.items()}

    def __init__(self, leaders: list[Leader], followers: list[Follower], pois: list[Poi],
                 max_steps, delta_time=1, render_mode=None):
        """
        https://pettingzoo.farama.org/api/parallel/

        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self._current_step = 0
        # self._agent_counters = {}
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.render_mode = render_mode

        self.leaders = leaders
        self.followers = followers
        self.pois = pois

        # def_obs_radius = np.sqrt(np.max(self.map_dimensions) ** 2 + np.max(self.map_dimensions) ** 2)
        # self.observation_radius = observation_radius if observation_radius else def_obs_radius

        # todo make possible to determine active from possible agents
        self._leaders = {f'{each_agent.name}': each_agent for each_agent in self.leaders}
        self._followers = {f'{each_agent.name}': each_agent for each_agent in self.followers}
        self._pois = {f'{each_agent.name}': each_agent for each_agent in self.pois}

        self.agents = [each_agent for each_agent in self.possible_agents]
        self.completed_agents = {}
        self.team_reward = 0

        self.reward_history = np.full(self.max_steps, -1)
        self.state_history = np.full(self.max_steps, -1)
        return

    @property
    def possible_agents(self):
        return list(self.agent_mapping.keys())

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name):
        """
        Gym spaces are defined and documented here: https://www.gymlibrary.dev/api/spaces/

        :param agent_name:
        :return:
        """
        return self.observation_spaces[agent_name]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_name):
        # Action space is desired velocity and heading
        return self.action_spaces[agent_name]

    def seed(self, seed=None):
        pass

    def state(self):
        """
        Returns the state.

        State returns a global view of the environment appropriate for centralized
        training decentralized execution methods like QMIX
        :return:
        """
        return self.state_history[self._current_step]

    def __render_rgb(self):
        render_resolution = (256, 256)
        render_bounds = (-5, 15)
        scaling = np.divide(render_resolution, render_bounds[1] - render_bounds[0])

        agent_colors = {'leader': [255, 0, 0], 'follower': [0, 255, 0], 'poi': [0, 0, 255]}
        agent_sizes = {'leader': 2, 'follower': 1, 'poi': 3}

        background_color = [255, 255, 255]
        line_color = [0, 0, 0]
        default_color = [128, 128, 128]
        default_size = 2
        num_lines = 10
        x_line_idxs = np.linspace(0, render_resolution[1], num=num_lines)
        y_line_idxs = np.linspace(0, render_resolution[0], num=num_lines)

        frame = np.full((render_resolution[0] + 1, render_resolution[1] + 1, 3), background_color)
        for each_line in x_line_idxs:
            each_line = int(each_line)
            frame[each_line] = line_color

        for each_line in y_line_idxs:
            each_line = int(each_line)
            frame[:, each_line] = line_color

        for agent_name in self.agents:
            agent = self.agent_mapping[agent_name]
            acolor = agent_colors.get(agent.type, default_color)
            asize = agent_sizes.get(agent.type, default_size)
            aloc = np.array(agent.location)

            scaled_loc = aloc - render_bounds[0]
            scaled_loc = np.multiply(scaled_loc, scaling)
            scaled_loc = np.rint(scaled_loc)
            scaled_loc = scaled_loc.astype(np.int)
            frame[
                scaled_loc[1] - asize: scaled_loc[1] + asize,
                scaled_loc[0] - asize: scaled_loc[0] + asize,
            ] = acolor
        frame = frame.astype(np.uint8)
        return frame

    def __Render_video(self):
        # todo implement video render
        return []

    def render(self, mode: str | None = 'human'):
        """
        Displays a rendered frame from the environment, if supported.

        Alternate render modes in the default environments are ‘rgb_array’ which returns a numpy array and
        is supported by all environments outside of classic, and ‘ansi’ which returns the strings printed
        (specific to classic environments).

        :param mode:
        :return:
        """
        if not mode:
            mode = self.render_mode

        match mode:
            case 'human':
                frame = self.__Render_video()
            case 'rgb_array':
                frame = self.__render_rgb()
            case _:
                frame = None
        return frame

    # def render(self, mode: Optional[Union[RenderMode, str]] = 'human'):
    #         if type(mode) == str:
    #             mode = RenderMode[mode]
    #
    #         if mode.value != RenderMode.none and self.render_mode.value == RenderMode.none:
    #             self.renderer = Renderer(
    #                 boids_colony=self.boids_colony,
    #                 poi_colony=self.poi_colony,
    #                 observation_manager=self.observation_manager,
    #                 **self.config["Renderer"]
    #             )
    #             self.render_mode = mode
    #
    #         self.renderer.renderFrame()


    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections or any other
        environment data which should not be kept around after the user is no longer using the environment.
        """
        pass

    def reset(self, seed: int | None = None, options: dict | None = None, **kwargs):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        And returns a dictionary of observations (keyed by the agent name).

        :param seed:
        :param options:
        :param kwargs:
        """
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self._current_step = 0

        _ = [each_agent.reset() for each_agent in self.leaders]
        _ = [each_agent.reset() for each_agent in self.followers]
        _ = [each_agent.reset() for each_agent in self.pois]

        # add all possible agents to the environment - agents are removed from the actors as they finish the task
        self.agents = [each_agent for each_agent in self.possible_agents]
        self.completed_agents = {}
        self.team_reward = 0

        observations = self.get_observations()
        return observations

    def get_observations(self):
        """
        Returns a dictionary of observations (keyed by the agent name).

        :return:
        """
        observations = {}
        for agent_name in self.agents:
            agent = self.agent_mapping[agent_name]
            agent_obs = agent.sense(self.agent_mapping.values())
            observations[agent_name] = agent_obs
        return observations

    def global_reward(self):
        return self.num_poi_observed()

    def difference_reward(self):
        # todo implement difference reward
        return

    def dpp_reward(self):
        # todo implement dpp reward
        return

    def num_poi_observed(self):
        return sum(poi.observed for poi in self.pois)

    # def __update_poi_state(self):
    #     for poi in self.pois:
    #         position = self.positions[poi.id]
    #         distances = calculateDistance(position, boids_colony_state.positions)
    #         num_observations = np.sum(distances <= self.observation_radius)
    #         if num_observations >= self.coupling:
    #             poi.observed = True
    #             # Get ids of swarm members that observed this poi
    #             observer_ids = np.nonzero(distances <= self.observation_radius)[0]
    #             poi.observation_list.append(observer_ids)
    #     return

    def done(self):
        all_obs = self.num_poi_observed() == len(self.pois)
        time_over = self._current_step >= self.max_steps
        val = any([all_obs, time_over])

        agent_dones = {each_agent: val for each_agent in self.agents}
        return agent_dones

    def step(self, actions):
        """
        actions of each agent are always the delta x, delta y

        obs, rew, terminated, truncated, info = par_env.step(actions)

        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # step leaders, followers, and pois
        # should not matter which order they are stepped in as long as dt is small enough
        for agent_name, each_action in actions.items():
            agent = self.agent_mapping[agent_name]
            # each_action[0] is dx
            # each_action[1] is dy
            new_loc = tuple(coord + vel * self.delta_time for coord, vel in zip(agent.location, each_action))
            agent.velocity = each_action
            agent.location = new_loc

        # Get all observations
        observations = self.get_observations()

        # Step forward and check if simulation is done
        # Update all agent dones with environment done
        self._current_step += 1
        dones = self.done()

        # Update infos and truncated for agents.
        # Not sure what would go here, but it seems to be expected by pettingzoo
        infos = {agent: {} for agent in self.agents}
        truncs = {agent: {} for agent in self.agents}

        # Calculate fitnesses
        rewards = {agent: 0.0 for agent in self.agents}
        rewards['team'] = 0.0
        if all(dones.values()):
            # todo properly calculate and use rewards at end of episode
            # calc_diff_rewards(self.boids_colony, self.poi_colony)
            self.completed_agents = {agent: 0 for agent in self.agents}
            # global_reward(self.poi_colony)
            self.team_reward = self.num_poi_observed()
            self.agents = []

        return observations, rewards, dones, truncs, infos
