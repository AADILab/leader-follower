from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment

if __name__ == '__main__':
    for coupling in [1,2,3,4,5]:
        for num_leaders in [1,2]:
            for num_pois in [1,2,3]:
                # Load in the config file
                config = loadConfig()

                # Compute configuration parameters
                if num_pois == 1:
                    poi_positions = [
                        [10,10]
                    ]
                elif num_pois == 2:
                    poi_positions = [
                        [10,10],
                        [10,20]
                    ]
                elif num_pois == 3:
                    poi_positions = [
                        [10,10],
                        [10,20],
                        [10,30]
                    ]
                if num_leaders == 1:
                    leader_positions = [
                        [30, 10]
                    ]
                elif num_leaders == 2:
                    leader_positions = [
                        [30, 10],
                        [30, 20]
                    ]

                # Set up the configuration file for this particular comparison                
                config["CCEA"]["config"]["BoidsEnv"]["config"]["BoidSpawner"]["leader_positions"] = leader_positions
                config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"] = num_leaders
                config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"] = poi_positions
                config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = coupling

                # Run each combination 20 times
                for _ in range(20):
                    config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
                    runExperiment(config)

                for _ in range(20):
                    config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
                    runExperiment(config)
                
                for _ in range(20):
                    config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
                    runExperiment(config)
