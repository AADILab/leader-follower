from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

import numpy as np

from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment

def turnToCoord(i, ax_length_x, ax_length_y):
    x = int(i / ax_length_y)
    y = i % ax_length_y
    return [x, y]

def getPosFromCoord(x, y, x_offset, y_offset, x_step, y_step):
    x_pos = x_offset + x*x_step
    y_pos = y_offset + y*y_step
    return [x_pos, y_pos]

def getPoiPosition(i, ax_length_x, ax_length_y):
    x_grid, y_grid = turnToCoord(i, ax_length_x, ax_length_y)
    x_offset = 10
    y_offset = 10
    x_step = 30
    y_step = 20
    x_pos, y_pos = getPosFromCoord(x_grid, y_grid, x_offset, y_offset, x_step, y_step)
    return [x_pos, y_pos]

def getPoiPositions(num_pois, ax_length_x, ax_length_y):
    """ Get the positions of followers from 0 to num_pois
    """
    poi_positions = []
    for i in range(num_pois):
        poi_positions.append(getPoiPosition(i, ax_length_x, ax_length_y))
    return poi_positions


if __name__ == '__main__':
    # Load the config file
    # Just load it once and modify it whenever we want to 
    # change it for a new experiment
    config = loadConfig()

    # 40x5 grid (but I'm only using 1x5)
    ax_length_x = 40
    ax_length_y = 5
    num_stat_runs = 3

    poi_positions = getPoiPositions(5, ax_length_x, ax_length_y)        
    
    leader_positions = np.array(poi_positions)
    leader_positions[:,0] += 20
    leader_positions = leader_positions.tolist()

    num_followers_per_leader_list = [3,4]
    distance_from_leader = 2.5
    xy_offset = float(np.sqrt((distance_from_leader**2)/2))

    config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = 5
    config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions

    for num_followers_per_leader in num_followers_per_leader_list:
        
        follower_positions = []

        for leader_position in leader_positions:
            for follower_id in range(num_followers_per_leader):
                if follower_id == 0:
                    follower_positions.append([leader_position[0]-xy_offset, leader_position[1]+xy_offset])
                elif follower_id == 1:
                    follower_positions.append([leader_position[0]+xy_offset, leader_position[1]+xy_offset])
                elif follower_id == 2:
                    follower_positions.append([leader_position[0]-xy_offset, leader_position[1]-xy_offset])
                elif follower_id == 3:
                    follower_positions.append([leader_position[0]+xy_offset, leader_position[1]-xy_offset])
                elif follower_id == 4:
                    follower_positions.append([leader_position[0], leader_position[1]+distance_from_leader])
                elif follower_id == 5:
                    follower_positions.append([leader_position[0]+distance_from_leader, leader_position[1]])
                elif follower_id == 6:
                    follower_positions.append([leader_position[0], leader_position[1]-distance_from_leader])
                elif follower_id == 7:
                    follower_positions.append([leader_position[0]-distance_from_leader, leader_position[1]])

        config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"] = follower_positions
        config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = len(follower_positions)
        config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = num_followers_per_leader

        print(type(follower_positions[0][0]))

        # import sys; sys.exit()

        # import sys; sys.exit()
        # Run each combination for the number of stat runs
        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
            print(config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"])
            runExperiment(config)

        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
            runExperiment(config)
        
        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
            runExperiment(config)
