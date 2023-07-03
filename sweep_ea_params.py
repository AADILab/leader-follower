from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment


if __name__ == '__main__':
    # Load the config file
    # Just load it once and modify it whenever we want to 
    # change it for a new experiment
    config = loadConfig()
    num_stat_runs = 3

    mutation_prob_list = [0.25  , 0.25,  0.5,  0.5]
    mutation_rate_list = [0.1   , 0.25,  0.1,  0.25]

    for mutation_prob, mutation_rate in zip(mutation_prob_list, mutation_rate_list):
        # config["CCEA"]["config"]["BoidsEnv"]["config"]["POIColony"]["observation_radius"] = obs_rad
        config["CCEA"]["mutation_probability"] = mutation_prob
        config["CCEA"]["mutation_rate"] = mutation_rate

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
