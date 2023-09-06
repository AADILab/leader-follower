import os
import sys
import time
from sys import exit
from copy import deepcopy
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

import numpy as np

from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment


def nextFollowerPositions(all_follower_positions):
    next_follower_positions = deepcopy(all_follower_positions[-4:])
    next_follower_positions[-4][1] += 40
    next_follower_positions[-3][1] += 40
    next_follower_positions[-2][1] += 40
    next_follower_positions[-1][1] += 40
    return next_follower_positions

def nextPoiPositions(all_poi_positions):
    next_poi_positions = deepcopy(all_poi_positions[-3:])
    next_poi_positions[-3][1] += 40
    next_poi_positions[-2][1] += 40
    next_poi_positions[-1][1] += 40
    return next_poi_positions


if __name__ == '__main__':
    # Compute follower positions
    # Start with 1. Add 9 more. Total of 10 follower pods
    all_follower_positions = [
        [20, 10],
        [25, 10],
        [20, 15],
        [25, 15]
    ]
    for i in range(9):
        next_follower_positions = nextFollowerPositions(all_follower_positions)
        all_follower_positions += next_follower_positions

    all_poi_positions = [
        [10, 10],
        [10, 20],
        [10, 30]
    ]
    for i in range(9):
        next_poi_positions = nextPoiPositions(all_poi_positions)
        all_poi_positions += next_poi_positions
    
    all_leader_positions = [
        [30, 10]
    ]
    for i in range(9):
        # print(all_leader_positions)
        next_leader_position = deepcopy(all_leader_positions[-1])
        # print(next_leader_position)
        next_leader_position[1]+=40
        # print(next_leader_position)
        all_leader_positions.append(next_leader_position)

    num_followers = 10*4
    num_leaders = 10

    map_y_lim = all_poi_positions[-1][1]+10

    config = loadConfig()
    # Set leaders, followers, pois
    config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"] = all_follower_positions
    config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = all_leader_positions
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_leaders
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
    config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = all_poi_positions
    config["CCEA"]["config"]["BoidsEnv"]["config"]["map_dimensions"]["y"] = map_y_lim

    # pp.pprint(all_follower_positions)
    # pp.pprint(all_poi_positions)
    # pp.pprint(all_leader_positions)

    runExperiment(config)
