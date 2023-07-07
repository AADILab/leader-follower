from lib.file_helper import loadConfig, saveConfig
# from lib.learn_helpers import runExperiment

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


# For naming here...
# Coords are just to track which leader/follower/poi is which
# Positions are for where these entities are actually placed in the map

def turnToCoord(i, ax_length):
    x = int(i / ax_length)
    y = i % ax_length
    return [x, y]

def getPosFromCoord(x, y, x_offset, y_offset, x_step, y_step):
    x_pos = x_offset + x*x_step
    y_pos = y_offset + y*y_step
    return [x_pos, y_pos]

def getLeaderPosition(i, ax_length):
    x_grid, y_grid = turnToCoord(i, ax_length)
    x_offset = 30
    y_offset = 10
    x_step = 30
    y_step = 10
    x_pos, y_pos = getPosFromCoord(x_grid, y_grid, x_offset, y_offset, x_step, y_step)
    return [x_pos, y_pos]

def getLeaderPositions(num_leaders, ax_length):
    """ Get the positions of leaders from 1 to num_leaders
        Just returns the position of the first leader if num_leaders is 1
    """
    leader_positions = []
    for i in range(num_leaders):
        leader_positions.append(getLeaderPosition(i, ax_length))
    return leader_positions

def getFollowerPosition(i, ax_length):
    x_grid, y_grid = turnToCoord(i, ax_length)
    x_offset = 20
    y_offset = 10
    x_step = 30
    y_step = 10
    x_pos, y_pos = getPosFromCoord(x_grid, y_grid, x_offset, y_offset, x_step, y_step)
    return [x_pos, y_pos]

def getFollowerPositions(num_followers, ax_length):
    """ Get the positions of followers from 0 to num_followers
    """
    follower_positions = []
    for i in range(num_followers):
        follower_positions.append(getFollowerPosition(i, ax_length))
    return follower_positions

def getPoiPosition(i, ax_length):
    x_grid, y_grid = turnToCoord(i, ax_length)
    x_offset = 10
    y_offset = 10
    x_step = 30
    y_step = 10
    x_pos, y_pos = getPosFromCoord(x_grid, y_grid, x_offset, y_offset, x_step, y_step)
    return [x_pos, y_pos]

def getPoiPositions(num_pois, ax_length):
    """ Get the positions of followers from 0 to num_pois
    """
    poi_positions = []
    for i in range(num_pois):
        poi_positions.append(getPoiPosition(i, ax_length))
    return poi_positions


# def setupLeaders(num_leaders: int):
#     leader_positions = []
#     for i_leader in range(num_leaders):
#         leader_pos = getLeaderPosition(i_leader)

if __name__ == '__main__':
    # 5x5 grid
    ax_length = 5
    num_stat_runs = 3



    # Load the config file
    # Just load it once and modify it whenever we want to 
    # change it for a new experiment
    config = loadConfig()

    for num_batch, i_group in enumerate([0,14,24,49]):
        num_groups = i_group+1

        # Set up leaders
        num_leaders = num_groups
        leader_positions = getLeaderPositions(num_leaders, ax_length)

        # Set up followers
        num_followers = num_groups
        follower_positions = getFollowerPositions(num_followers, ax_length)

        # Set up pois
        num_pois = num_groups
        poi_positions = getPoiPositions(num_pois, ax_length)

        # Set up the configuration           
        config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions
        config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_leaders
        config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"] = follower_positions
        config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
        config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions

        # set number of generations and coupling properly
        config["num_generations"] = 1000
        config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = 1
        config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["observation_radius"] = 10000 # Dense
        config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_G"] = "ContinuousObsRad" #Informative

        # runExperiment(config)
        # import sys; sys.exit()

        # Run each combination n times
        for i in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
            trial_num = num_batch*3*num_stat_runs+i
            # Save this config
            saveConfig(config=config, computername="experiment_1a", trial_num=str(trial_num))

        for i in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
            trial_num = num_batch*3*num_stat_runs+num_stat_runs+i
            saveConfig(config=config, computername="experiment_1a", trial_num=str(trial_num))

        for i in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
            trial_num = num_batch*3*num_stat_runs+2*num_stat_runs+i
            saveConfig(config=config, computername="experiment_1a", trial_num=str(trial_num))