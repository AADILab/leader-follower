import sys
from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment

# Load in config
config = loadConfig()

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
