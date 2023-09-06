from lib.file_helper import loadConfig, saveConfig
# from lib.learn_helpers import runExperiment
import numpy as np

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


# For naming here...
# Coords are just to track which leader/follower/poi is which
# Positions are for where these entities are actually placed in the map

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

    # need a name for this experiment
    experiment_name = "coupling_1_5leaders"
    coupling = 1

    # 40x5 grid (but I'm only using 1x5)
    ax_length_x = 40
    ax_length_y = 5
    num_stat_runs = 10

    poi_positions = getPoiPositions(5, ax_length_x, ax_length_y)        
    
    leader_positions = np.array(poi_positions)
    leader_positions[:,0] += 20
    leader_positions = leader_positions.tolist()

    num_followers_per_leader_list = [1,2,3,4]
    distance_from_leader = 2.5
    xy_offset = float(np.sqrt((distance_from_leader**2)/2))

    config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions
    config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = 5
    config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions

    for follower_per_leader_ind, num_followers_per_leader in enumerate(num_followers_per_leader_list):
        
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
        config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = coupling
        config["CCEA"]["num_workers"] = 20
        config["num_generations"] = 2000

        print(type(follower_positions[0][0]))

        # import sys; sys.exit()

        # import sys; sys.exit()
        # Run each combination for the number of stat runs
        for i in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
            # print(config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"])
            trial_num = follower_per_leader_ind*num_stat_runs*3 + i
            # Save this config
            saveConfig(config=config, computername=experiment_name, trial_num=str(trial_num))

        for i in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
            trial_num = follower_per_leader_ind*num_stat_runs*3 + num_stat_runs + i
            saveConfig(config=config, computername=experiment_name, trial_num=str(trial_num))
        
        for i in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
            trial_num = follower_per_leader_ind*num_stat_runs*3 + 2*num_stat_runs+i
            saveConfig(config=config, computername=experiment_name, trial_num=str(trial_num))


# if __name__ == '__main__':
#     # 5x5 grid
#     ax_length = 5
#     num_stat_runs = 10

#     experiment_name = "coupling_5lf_1c"
#     coupling = 1
#     # Increasing the number of followers as a sweep

#     # Load the config file
#     # Just load it once and modify it whenever we want to 
#     # change it for a new experiment
#     config = loadConfig()

#     num_groups = 5

#     # Set up leaders
#     num_leaders = num_groups
#     leader_positions = getLeaderPositions(num_leaders, ax_length)

#     # Set up followers
#     num_followers = num_groups
#     follower_positions = getFollowerPositions(num_followers, ax_length)

#     # Set up pois
#     num_pois = num_groups
#     poi_positions = getPoiPositions(num_pois, ax_length)

#     # Set up the configuration           
#     config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions
#     config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_leaders
#     config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"] = follower_positions
#     config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
#     config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions

#     # set number of generations and coupling properly
#     config["num_generations"] = 2000
#     config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = coupling
#     config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["observation_radius"] = 1000 # Dense
#     config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_G"] = "ContinuousObsRadLastStep" # UnInformative

#     # runExperiment(config)
#     # import sys; sys.exit()

#     # Run each combination n times
#     for i in range(num_stat_runs):
#         config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
#         trial_num = 3*num_stat_runs+i
#         # Save this config
#         saveConfig(config=config, computername=experiment_name, trial_num=str(trial_num))

#     for i in range(num_stat_runs):
#         config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
#         trial_num = 3*num_stat_runs+num_stat_runs+i
#         saveConfig(config=config, computername=experiment_name, trial_num=str(trial_num))

#     for i in range(num_stat_runs):
#         config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
#         trial_num = 3*num_stat_runs+2*num_stat_runs+i
#         saveConfig(config=config, computername=experiment_name, trial_num=str(trial_num))