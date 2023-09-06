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
    y_step = 10
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

    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = 5
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = 5
    # config["CCEA"]["config"][]

    # num_stat_runs = 3
    distances = [10,20,30,40]
    num_steps = [50,100,150,200]

    # 40x5 grid (but I'm only using 1x5)
    ax_length_x = 40
    ax_length_y = 5
    num_stat_runs = 5

    poi_positions = getPoiPositions(5, ax_length_x, ax_length_y)

    for distance, num_step in zip(distances, num_steps):
        
        follower_positions = np.array(poi_positions)
        follower_positions[:,0] += distance
        follower_positions = follower_positions.tolist()

        leader_positions = np.array(follower_positions)
        leader_positions[:,0] += 10
        leader_positions = leader_positions.tolist()

        config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions
        config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"] = follower_positions
        config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions

        # runExperiment(config)

        # Run each combination for the number of stat runs
        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
            runExperiment(config)

        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
            runExperiment(config)
        
        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
            runExperiment(config)
