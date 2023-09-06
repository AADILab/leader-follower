from lib.file_helper import loadConfig
from lib.learn_helpers import runExperiment

if __name__ == '__main__':
    # Run each experiment 10 times
    for _ in range(10):
        config = loadConfig()
        config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
        runExperiment(config)

    for _ in range(10):
        config = loadConfig()
        config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
        runExperiment(config)
    
    for _ in range(10):
        config = loadConfig()
        config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
        runExperiment(config)
