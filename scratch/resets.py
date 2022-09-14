from lib.learn_helpers import loadConfig
from lib.boids_env import BoidsEnv
import time
import numpy as np

config = loadConfig()

env_config = config["CCEA"]["config"]["BoidsEnv"]

env = BoidsEnv(**env_config)

for i in range(10):
    env.reset(seed=0)
    for _ in range(60):
        env.step({agent: np.zeros(2) for agent in env.agents})
        env.render()
        time.sleep(1/60)
