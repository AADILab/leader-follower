from lib.file_helper import loadConfig
import numpy as np

config = loadConfig()

# Seed the random number generator
np.random.seed(0)

# Set the map x,y
map_x = 100
map_y = 100

# Get the poi positions
poi_positions = np.random

# config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"


#         # Set up the configuration           
#         config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions
#         config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_leaders
#         config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"] = follower_positions
#         config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
#         config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions
