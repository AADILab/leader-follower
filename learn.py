import sys
import numpy as np
from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment
from sys import exit

# Try same task with different ratios of learners to followers
config = loadConfig()

team_size = 15
num_learners =  np.arange(team_size)+1

for num_learn in num_learners:
    for _ in range(3):
        config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_learn
        config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = team_size - num_learn
        runExperiment(config)
exit()

config = loadConfig()
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = 5
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = 0
runExperiment(config)

for coupling in [1,2,3,4]:
    config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = coupling
    runExperiment(config)
exit()

# Try these experiments again but with 2 leaders.
# The difficulty in learning may come from too many agents, not enough tasks
config = loadConfig()
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = 2
for num_units in [8,10,12,14]:
    config["CCEA"]["nn_hidden"] = [num_units]
    runExperiment(config)

    config["CCEA"]["nn_hidden"] = [num_units, num_units]
    runExperiment(config)

# Do a narrow sweep with 3 leaders and more nuerons
config = loadConfig()
for num_units in [12,13,14,15,16]:
    config["CCEA"]["nn_hidden"] = [num_units, num_units]
    runExperiment(config)

exit()


# Experiment with adding followers.
for num_followers in [0,1,2,3,4]:
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
    config["Notes"] = str(num_followers)+" followers. 1 leader. 1 coupling."
    runExperiment(config)

# Raise the coupling to 2
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 2

# Experiment with adding followers.
for num_followers in [1,2,3,4]:
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
    config["Notes"] = str(num_followers)+" followers. 1 leader. 2 coupling."
    runExperiment(config)

# Raise the coupling to 3
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 3

# Experiment with adding followers.
for num_followers in [2,3,4]:
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
    config["Notes"] = str(num_followers)+" followers. 1 leader. 3 coupling."
    runExperiment(config)

# Raise the coupling to 4
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 4

# Experiment with adding followers.
for num_followers in [3,4]:
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
    config["Notes"] = str(num_followers)+" followers. 1 leader. 4 coupling."
    runExperiment(config)

# Raise the coupling to 5
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 5
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = 4
config["Notes"] = "4 followers. 1 leader. 5 coupling."
runExperiment(config)

sys.exit()
