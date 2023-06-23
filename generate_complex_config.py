from lib.file_helper import loadConfig
import numpy as np
import pprint
from lib.learn_helpers import runExperiment
pp = pprint.PrettyPrinter()

config = loadConfig()

# Seed the random number generator
np.random.seed(0)

# Set the map x,y
map_x = 100
map_y = 100


num_leaders = 5
num_followers = 15
num_pois = 5

# poi_positions = []
# for 

poi_positions = np.random.rand(5,2)
poi_positions[:,0]*=map_x
poi_positions[:,1]*=map_y
poi_positions = poi_positions.tolist()

leader_positions = np.random.rand(5,2)
leader_positions[:,0] *= map_x
leader_positions[:,1] *= map_y
leader_positions = leader_positions.tolist()

follower_positions = np.random.rand(15,2)
follower_positions[:,0] *= map_x
follower_positions[:,1] *= map_y
follower_positions = follower_positions.tolist()

coupling = 3
observation_radius = 1000

config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions
config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"] = follower_positions
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_leaders
config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers

config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions

config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = coupling
config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["observation_radius"] = observation_radius

config["CCEA"]["config"]["BoidsEnv"]["config"]["map_dimensions"]["x"] = map_x
config["CCEA"]["config"]["BoidsEnv"]["config"]["map_dimensions"]["y"] = map_y

runExperiment(config)
