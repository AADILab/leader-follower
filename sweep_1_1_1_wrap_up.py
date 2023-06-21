from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment

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

def getLeaderPosition(i, ax_length_x, ax_length_y):
    x_grid, y_grid = turnToCoord(i, ax_length_x, ax_length_y)
    x_offset = 30
    y_offset = 10
    x_step = 30
    y_step = 10
    x_pos, y_pos = getPosFromCoord(x_grid, y_grid, x_offset, y_offset, x_step, y_step)
    return [x_pos, y_pos]

def getLeaderPositions(num_leaders, ax_length_x, ax_length_y):
    """ Get the positions of leaders from 1 to num_leaders
        Just returns the position of the first leader if num_leaders is 1
    """
    leader_positions = []
    for i in range(num_leaders):
        leader_positions.append(getLeaderPosition(i, ax_length_x, ax_length_y))
    return leader_positions

def getFollowerPosition(i, ax_length_x, ax_length_y):
    x_grid, y_grid = turnToCoord(i, ax_length_x, ax_length_y)
    x_offset = 20
    y_offset = 10
    x_step = 30
    y_step = 10
    x_pos, y_pos = getPosFromCoord(x_grid, y_grid, x_offset, y_offset, x_step, y_step)
    return [x_pos, y_pos]

def getFollowerPositions(num_followers, ax_length_x, ax_length_y):
    """ Get the positions of followers from 0 to num_followers
    """
    follower_positions = []
    for i in range(num_followers):
        follower_positions.append(getFollowerPosition(i, ax_length_x, ax_length_y))
    return follower_positions

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


# def setupLeaders(num_leaders: int):
#     leader_positions = []
#     for i_leader in range(num_leaders):
#         leader_pos = getLeaderPosition(i_leader)

if __name__ == '__main__':
    # 40x5 grid
    ax_length_x = 40
    ax_length_y = 5
    num_stat_runs = 3



    # Load the config file
    # Just load it once and modify it whenever we want to 
    # change it for a new experiment
    config = loadConfig()

    """Compute the baseline D for all of the experiments I already ran"""
    for i_group in [0,4,9,14,19,24]:
    # for i_group in [49, 99]:
    # for i_group in [199, 399]:
        num_groups = i_group+1

        # Set up leaders
        num_leaders = num_groups
        leader_positions = getLeaderPositions(num_leaders, ax_length_x, ax_length_y)

        # Set up followers
        num_followers = num_groups
        follower_positions = getFollowerPositions(num_followers, ax_length_x, ax_length_y)

        # Set up pois
        num_pois = num_groups
        poi_positions = getPoiPositions(num_pois, ax_length_x, ax_length_y)

        # pp.pprint(leader_positions)
        # pp.pprint(follower_positions)
        # pp.pprint(poi_positions)

        # Set up the configuration           
        config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions
        config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_leaders
        config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"] = follower_positions
        config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
        config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions

        # runExperiment(config)
        # import sys; sys.exit()

        # Run each combination 10 times
        # for _ in range(num_stat_runs):
        #     config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
        #     runExperiment(config)

        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
            runExperiment(config)
        
        # for _ in range(num_stat_runs):
        #     config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
        #     runExperiment(config)
        
        # for _ in range(num_stat_runs):
        #     config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "Zero"
        #     runExperiment(config)

        # x,y = turnToCoord(i=10, ax_length=5)

        # print(x,y)

    """Compute the results for 50 and 100 triplets. Need to rerun for G and DFollow so we compute up to 100 gens"""
    for i_group in [49, 99]:
    # for i_group in [49, 99]:
    # for i_group in [199, 399]:
        num_groups = i_group+1

        # Set up leaders
        num_leaders = num_groups
        leader_positions = getLeaderPositions(num_leaders, ax_length_x, ax_length_y)

        # Set up followers
        num_followers = num_groups
        follower_positions = getFollowerPositions(num_followers, ax_length_x, ax_length_y)

        # Set up pois
        num_pois = num_groups
        poi_positions = getPoiPositions(num_pois, ax_length_x, ax_length_y)

        # pp.pprint(leader_positions)
        # pp.pprint(follower_positions)
        # pp.pprint(poi_positions)

        # Set up the configuration           
        config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions
        config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_leaders
        config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["follower_positions"] = follower_positions
        config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"] = num_followers
        config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions

        # runExperiment(config)
        # import sys; sys.exit()

        # Run each combination 10 times
        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
            runExperiment(config)

        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
            runExperiment(config)
        
        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
            runExperiment(config)
        
        # for _ in range(num_stat_runs):
        #     config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "Zero"
        #     runExperiment(config)

        # x,y = turnToCoord(i=10, ax_length=5)

        # print(x,y)

