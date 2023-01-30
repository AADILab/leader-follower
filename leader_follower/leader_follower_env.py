import functools
import pickle
from pathlib import Path

import numpy as np

from leader_follower.agent import Poi, Leader, Follower, AgentType


class LeaderFollowerEnv:

    metadata = {'render_modes': ['human', 'rgb_array', 'none'], 'name': 'leader_follower_environment'}

    @property
    def agent_mapping(self):
        return self.leaders | self.followers | self.pois

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
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.render_mode = render_mode

        # todo make possible to determine active from possible agents
        self.leaders = {f'{each_agent.name}': each_agent for each_agent in leaders}
        self.followers = {f'{each_agent.name}': each_agent for each_agent in followers}
        self.pois = {f'{each_agent.name}': each_agent for each_agent in pois}

        self.agents = [each_agent for each_agent in self.possible_agents]
        self.completed_agents = {}
        self.team_reward = 0
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
        # todo  store state as a matrix in environment rather than individually in agents
        #       env is the state and agents are how the updates are calculated based on current state
        #       note that this may imply non-changing set of agents
        pass

    def save_environment(self, base_dir, tag=''):
        # todo  use better methods of saving than pickling
        # https://docs.python.org/3/library/pickle.html#pickling-class-instances
        # https://stackoverflow.com/questions/37928794/which-is-faster-for-load-pickle-or-hdf5-in-python
        # https://marshmallow.readthedocs.io/en/stable/
        # https://developers.google.com/protocol-buffers
        # https://developers.google.com/protocol-buffers/docs/pythontutorial
        if tag != '':
            tag = f'_{tag}'

        save_path = Path(base_dir, f'leader_follower_env{tag}.pkl')
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        self.reset()
        with open(save_path, 'wb') as save_file:
            pickle.dump(self, save_file, pickle.HIGHEST_PROTOCOL)
        return save_path

    @staticmethod
    def load_environment(load_path):
        with open(load_path, 'rb') as load_file:
            env = pickle.load(load_file)
        return env

    def __render_rgb(self):
        # todo set based on min/max agent locations
        render_resolution = (512, 512)
        render_bounds = (-5, 55)
        scaling = np.divide(render_resolution, render_bounds[1] - render_bounds[0])

        agent_colors = {AgentType.Learner: [255, 0, 0], AgentType.Actor: [0, 255, 0], AgentType.Static: [0, 0, 255]}
        agent_sizes = {AgentType.Learner: 2, AgentType.Actor: 1, AgentType.Static: 3}

        background_color = [255, 255, 255]
        line_color = [0, 0, 0]
        default_color = [128, 128, 128]
        default_size = 2
        num_lines = 10
        x_line_idxs = np.linspace(0, render_resolution[1], num=num_lines)
        y_line_idxs = np.linspace(0, render_resolution[0], num=num_lines)

        frame = np.full((render_resolution[0] + 1, render_resolution[1] + 1, 3), background_color)

        # draw a grid over the frame
        for each_line in x_line_idxs:
            each_line = int(each_line)
            frame[each_line] = line_color

        for each_line in y_line_idxs:
            each_line = int(each_line)
            frame[:, each_line] = line_color

        # place the agents in the frame based on the sizes and colors specified in agent_colors and agent_sizes
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

    def __render_video(self):
        # todo implement video render
        pass

    def render(self, mode: str | None = None):
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
                # frame = self.__render_video()
                frame = None
            case 'rgb_array':
                frame = self.__render_rgb()
            case _:
                frame = None
        return frame


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

        # add all possible agents to the environment - agents are removed from the self.agents as they finish the task
        self.agents = self.possible_agents[:]
        self._current_step = 0

        _ = [each_agent.reset() for each_agent in self.leaders.values()]
        _ = [each_agent.reset() for each_agent in self.followers.values()]
        _ = [each_agent.reset() for each_agent in self.pois.values()]

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
        agent_refs = list(self.agent_mapping.values())
        for agent_name in self.agents:
            agent = self.agent_mapping[agent_name]

            # agent_obs = agent.sense(state)
            agent_obs = agent.sense(agent_refs)
            observations[agent_name] = agent_obs
        return observations

    def get_actions(self):
        observations = self.get_observations()
        actions = self.get_actions_from_observations(observations)
        return actions

    def get_actions_from_observations(self, observations):
        """
        Returns a dictionary of actions (keyed by the agent name).

        :return:
        """
        actions = {}
        for agent_name in self.agents:
            agent = self.agent_mapping[agent_name]
            agent_obs = observations[agent_name]

            agent_action = agent.get_action(agent_obs)
            actions[agent_name] = agent_action
        return actions

    def num_poi_observed(self):
        return sum(poi.observed for poi in self.pois.values())

    def done(self):
        all_obs = self.num_poi_observed() == len(self.pois.values())
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
            agent.location = new_loc

        # todo track actions and observations in step function, not when functions called in agent implementation
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
