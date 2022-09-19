import sys
from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment

# Load in config
config = loadConfig()

# Run starting experiment which is simple. No followers, no coupling
config["CCEA"]["num_workers"] = 10
config["Notes"] = "Starter Experiment. 5 leaders. No followers, no coupling"
runExperiment(config)

# Add followers, add coupling
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = 5
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 2
config["Notes"] = "5 Followers, 2 coupling"
runExperiment(config)

# Add followers. Increase coupling
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = 10
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 3
config["Notes"] = "10 followers, 3 coupling"
runExperiment(config)

# Add followers. Increase coupling
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = 15
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 4
config["Notes"] = "15 followers, 4 coupling"
runExperiment(config)

# Add followers. Increase coupling
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = 20
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 5
config["Notes"] = "20 followers, 5 coupling"
runExperiment(config)

# Ramp up coupling. No follower increase
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 10
config["Notes"] = "20 followers, 10 coupling"
runExperiment(config)

# Reset to base config
config = loadConfig()
config["CCEA"]["num_workers"] = 10

# Experiment with adding more learners
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = 20
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 5
for num_leaders in range(5):
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_leaders
    config["Notes"] = str(num_leaders)+" learner. 20 followers. 5 coupling"
    runExperiment(config)

sys.exit()
