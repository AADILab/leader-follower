import sys; sys.path.append("/home/gonzaeve/boids/leader-follower")
from leaderfollower.file_helper import loadConfigDir, saveConfig

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

if __name__ == '__main__':
    # Load the config file
    config = loadConfigDir(config_dir="configs/alpha/25_lfp/25_lfp_wide_straight_a.yaml")
    config["num_generations"] = 500

    num_stat_runs = 20
    experiment_name = "experiment_9a_25leaders"

    RUN_G = True
    RUN_D = True
    RUN_Df = True

    # Run each combination n times
    num_batch = 0
    trial_num = 0

    if RUN_G:
        for i in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
            # Save this config
            saveConfig(config=config, computername=experiment_name, trial_num=str(trial_num), folder_save=True)
            trial_num += 1

    if RUN_D:
        for i in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
            saveConfig(config=config, computername=experiment_name, trial_num=str(trial_num), folder_save=True)
            trial_num += 1

    if RUN_Df:
        for i in range(num_stat_runs):
            config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
            saveConfig(config=config, computername=experiment_name, trial_num=str(trial_num), folder_save=True)
            trial_num += 1
