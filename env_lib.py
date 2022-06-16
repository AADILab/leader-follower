import functools
import enum
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import from_parallel
import numpy as np

from boids_manager import BoidsManager
from renderer import Renderer
from learning_module_lib import LearningModule

ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}

class OBSERVATION(enum.IntEnum):
    GOAL_AND_CENTROID = 0

class REWARD(enum.IntEnum):
    DISTANCE_TO_GOAL = 0

def env():
    '''
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env():
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = BoidsEnv()
    env = from_parallel(env)
    return env


class BoidsEnv(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, num_leaders = 2, num_followers = 10, FPS = 60, positions = None, r_ind = None, learning_module: LearningModule = None):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        # Only leaders are included in self.possible_agents
        # because they are the learners
        self.possible_agents = ["leader_" + str(r) for r in np.arange(num_leaders)+1]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(np.arange(num_leaders)+1)))
        print("self.agent_name_mapping:\n", self.agent_name_mapping)

        map_size = np.array([100,100])
        rs = (2,3,5)
        self.bm = BoidsManager(num_leaders=num_leaders, num_followers=num_followers, max_velocity=2.5, max_angular_velocity=np.pi*0.5, radius_repulsion=rs[0], radius_orientation=rs[1], radius_attraction=rs[2], map_size=map_size, ghost_density=10, dt=1/FPS, positions=positions)
        self.renderer = Renderer(num_leaders, num_followers, map_size, pixels_per_unit=5, radii = rs, r_ind=r_ind)

        # Setup learning module
        self.lm = self.setupLearningModule(learning_module)

    def setupLearningModule(self, learning_module):
        if learning_module is None:
            return LearningModule(goal_locations = np.array([self.bm.map_size])/2)
        else:
            return learning_module

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.lm.observation_space()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Action space is velocity and desired heading
        # Desired heading is relative to agent's own reference frame
        return Box(
            low=np.array([0, -np.pi],
            high=np.array[self.max_velocity, np.pi]),
            dtype=np.float32
        )

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        self.renderer.renderFrame(self.bm.positions, self.bm.headings)

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self):
        '''
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        '''
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: NONE for agent in self.agents}
        return observations

    def getObservations(self):
        observations_lm = self.lm.getObservations(self.bm)
        observations = {}
        for agent_id, agent_observation in observations_lm.items():
            observations[self.possible_agents[agent_id]] = agent_observation
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

        # Step forward all of the follower boids.
        self.bm.step()

        # Get the observations of the leader boids
        observations = self.getObservations()

        # Get rewards for leaders
        rewards = self.lm.getRewards(self.bm, {})

        dones = {}
        infos = {}

        # Return observations of leader boids AKA "agents"
        return observations, rewards, dones, infos

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        rewards[self.agents[0]], rewards[self.agents[1]] = REWARD_MAP[(actions[self.agents[0]], actions[self.agents[1]])]

        self.num_moves += 1
        env_done = self.num_moves >= NUM_ITERS
        dones = {agent: env_done for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {self.agents[i]: int(actions[self.agents[1 - i]]) for i in range(len(self.agents))}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, dones, infos

