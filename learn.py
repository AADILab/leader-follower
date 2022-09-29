import sys
from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment
from sys import exit


for mut_rate in [0.1, 0.15, 0.2, 0.25, 0.3]:
    for mut_prob in [0.15, 0.2, 0.25, 0.3]:
        for nn_hidden in [10,11,12,13,14,15]:
            for _ in range(3):
                config = loadConfig()
                config["CCEA"]["nn_hidden"] = nn_hidden
                config["CCEA"]["mutation_probability"] = mut_prob
                config["CCEA"]["mutation_rate"] = mut_rate
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
