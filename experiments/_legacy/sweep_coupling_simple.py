from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment


if __name__ == '__main__':
    # Load the config file
    # Just load it once and modify it whenever we want to 
    # change it for a new experiment
    config = loadConfig()
    num_stat_runs = 10

    for coupling in [1,2,3]:
        config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["coupling"] = coupling

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
        
        for _ in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "Zero"
            runExperiment(config)
