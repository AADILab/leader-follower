import copy
import functools
import pickle
from pathlib import Path

import numpy as np
import pygame

from leader_follower.agent import Poi, Leader, Follower, AgentType


class LeaderFollowerEnv:

    metadata = {'render_modes': ['human', 'rgb_array', 'none'], 'render_fps': 4, 'name': 'leader_follower_environment'}

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

        # todo make possible to determine active from possible agents
        self.leaders = {f'{each_agent.name}': each_agent for each_agent in leaders}
        self.followers = {f'{each_agent.name}': each_agent for each_agent in followers}
        self.pois = {f'{each_agent.name}': each_agent for each_agent in pois}

        self.agents = [each_agent for each_agent in self.possible_agents]
        self.completed_agents = {}
        self.team_reward = 0

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        # The size of the PyGame window
        self.render_bound = 100
        self.window_size = 512
        self.window = None
        self.clock = None
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

    def __render_frame(self):
        self.render_bound = 50
        self.window_size = 512
        self.window = None
        self.clock = None

        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # The size of a single grid square in pixels
        pix_square_size = (self.window_size / self.render_bound)

        leader_color = (255, 0, 0)
        follower_color = (0, 0, 255)
        obs_poi_color = (0, 255, 0)
        non_obs_poi_color = (0, 0, 0)
        line_color = (192, 192, 192)

        # draw some gridlines
        for x in range(self.render_bound + 1):
            pygame.draw.line(
                canvas, line_color, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=1,
            )
            pygame.draw.line(
                canvas, line_color, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=1,
            )

        for name, agent in self.leaders.items():
            location = np.array(agent.location)
            pygame.draw.rect(
                canvas, leader_color, pygame.Rect(pix_square_size * location, (pix_square_size, pix_square_size))
            )

        for name, agent in self.followers.items():
            location = np.array(agent.location)
            pygame.draw.circle(canvas, follower_color, (location + 0.5) * pix_square_size, pix_square_size / 1.5)

        for name, agent in self.pois.items():
            # different colors to distinguish if the poi is captured
            location = np.array(agent.location)
            agent_color = obs_poi_color if agent.observed else non_obs_poi_color
            pygame.draw.circle(canvas, agent_color, (location + 0.5) * pix_square_size, pix_square_size / 1.5)

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array
            np_frame = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
            return np_frame
        return

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
                frame = self.__render_frame()
            case 'rgb_array':
                frame = self.__render_frame()
                # frame = self.__render_rgb()
            case _:
                frame = None
        return frame


    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections or any other
        environment data which should not be kept around after the user is no longer using the environment.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        return

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

    def observed_pois(self):
        observed = [poi for name, poi in self.pois.items() if poi.observed]
        return observed

    def done(self):
        all_obs = len(self.observed_pois()) == len(self.pois.values())
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
            self.completed_agents = self.calc_diff_rewards(remove_followers=False)
            self.team_reward = self.calc_global()
            self.agents = []

        return observations, rewards, dones, truncs, infos

    # Reward functions
    def calc_global(self):
        # todo  value based on distance to poi rather than if observed
        observed_pois = self.observed_pois()
        value_pois = sum(each_poi.value for each_poi in observed_pois)
        total_value = sum(each_poi.value for name, each_poi in self.pois.items())
        reward = value_pois / total_value
        return reward

    def assign_followers(self):
        # todo change difference calculation to use this function
        assigned_followers = {
            leader_name: [leader_name]
            for leader_name, leader in self.leaders.items()
        }
        assigned_followers["leader_null"] = []

        follower_influences = {
            follower_name: follower.influence_counts()[0]
            for follower_name, follower in self.followers.items()
        }

        for follower_name, counts in follower_influences.items():
            for idx, name in enumerate(counts[0]):
                if not name.startswith('leader'):
                    counts[1][idx] = -1
            if len(counts[1]) == 0:
                max_influencer = "leader_null"
            else:
                max_idx = np.argmax(counts[1])
                max_influencer = counts[0][max_idx]
            assigned_followers[max_influencer].append(follower_name)
        return assigned_followers

    def calc_diff_rewards(self, remove_followers=False):
        """

        :param remove_followers:
        :return:
        """
        # todo  add tracking of calls to calc_global
        assigned_followers = {
            leader_name: []
            for leader_name, leader in self.leaders.items()
        }
        assigned_followers["leader_null"] = []

        if remove_followers:
            follower_influences = {
                follower_name: follower.influence_counts()[0]
                for follower_name, follower in self.followers.items()
            }

            for follower_name, counts in follower_influences.items():
                for idx, name in enumerate(counts[0]):
                    if not name.startswith('leader'):
                        counts[1][idx] = -1
                if len(counts[1]) == 0:
                    max_influencer = "leader_null"
                else:
                    max_idx = np.argmax(counts[1])
                    max_influencer = counts[0][max_idx]
                assigned_followers[max_influencer].append(follower_name)

        # todo  explore: if multiple agents are individually capable of observing a poi, neither receives a reward
        global_reward = self.calc_global()
        difference_rewards = {"G": global_reward}
        for leader, removed_agents in assigned_followers.items():
            removed_agents.append(leader)

            poi_copy = copy.deepcopy(self.pois)
            for poi_name, poi in self.pois.items():
                pruned_history = []
                for observation_step in poi.observation_history:
                    pruned_step = [
                        agent
                        for agent in observation_step
                        if agent[0].name not in removed_agents
                    ]
                    pruned_history.append(pruned_step)
                poi.observation_history = pruned_history
            difference_global = self.calc_global()
            difference_rewards[leader] = global_reward - difference_global
            self.pois = poi_copy
        return difference_rewards

    def calc_dpp_n(self, agent_names, n):
        """
        Calculate the reward in the counterfactual case where there are n copies of the given agent.
        This is equivalent to multiplying the agents true value by n.

        agent_names is an iterable and copy each agent in the iterable n times

        Note that when n = 0, this is effectively the difference reward.

        :param agent_names:
        :param n:
        :return:
        """
        orig_vals = {}
        for each_name in agent_names:
            agent = self.agent_mapping[each_name]
            orig_agent_val = agent.value
            agent.value = orig_agent_val * n
            orig_vals[each_name] = orig_agent_val

        # todo verify calculation of dpp_n
        reweighted_reward = self.calc_global()
        for each_name, each_orig_val in orig_vals.items():
            agent = self.agent_mapping[each_name]
            agent.value = each_orig_val
        return reweighted_reward

    def calc_dpp(self, remove_followers=False):
        """
        Calculate D++ rewards for each rover

        DPP pseudocode

        1. calculate D++^{-1}
        2. calculate D++^{total_agents - 1}
        3. if D++^{total_agents - 1} <= D++^{-1}
        4.  return D++^{total_agents - 1}
        5. else:
        6.  n := 0
        7.  repeat:
        8.      n += 1
        9.      calculate D++^{n}
        10.     if calculate D++^{n} > D++^{n - 1}
        11.         return D++^{n}
        12. until n <= total_agents - 1
        13. return D++^{-1}

        :param remove_followers:
        :return dpp_rewards: Numpy array containing each rover's D++ reward
        """
        num_agents = len(self.leaders)
        dpp_rewards = {name: 0 for name, agent in self.leaders.items()}
        assigned = self.assign_followers()
        for leader_name, leader in self.leaders.items():
            # add assigning followers to the leader before calculating the dpp reward for the given agent
            agent_names = [leader_name]
            if remove_followers:
                follower_names = assigned[leader_name]
                agent_names.extend(follower_names)

            dpp_min = self.calc_dpp_n(agent_names=agent_names, n=0)
            dpp_max = self.calc_dpp_n(agent_names=agent_names, n=num_agents - 1)

            if dpp_max <= dpp_min:
                dpp_rewards[leader_name] = dpp_max
            else:
                dpp_rewards[leader_name] = dpp_min
                prev_dpp_n = dpp_min
                for val_n in range(1, num_agents - 1):
                    dpp_n = self.calc_dpp_n(agent_names=[leader_name], n=val_n)
                    if dpp_n > prev_dpp_n:
                        dpp_rewards[leader_name] = dpp_n
                        break
                    prev_dpp_n = dpp_n
        return dpp_rewards
