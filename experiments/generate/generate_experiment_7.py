from leaderfollower.file_helper import loadConfigDir, saveConfig

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

if __name__ == '__main__':
    # Load the config file
    config = loadConfigDir(config_dir="configs/alpha/corners_count_leaders.yaml")
    config["num_generations"] = 1000

    num_stat_runs = 20
    experiment_name = "experiment_7"

    # Run each combination n times
    num_batch = 0
    for i in range(num_stat_runs):
        config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "G"
        trial_num = num_batch*3*num_stat_runs+i
        # Save this config
        saveConfig(config=config, computername="experiment_7", trial_num=str(trial_num), folder_save=True)

    for i in range(num_stat_runs):
        config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "D"
        trial_num = num_batch*3*num_stat_runs+num_stat_runs+i
        saveConfig(config=config, computername="experiment_7", trial_num=str(trial_num), folder_save=True)

    for i in range(num_stat_runs):
        config["CCEA"]["config"]["BoidsEnv"]["config"]["FitnessCalculator"]["which_D"] = "DFollow"
        trial_num = num_batch*3*num_stat_runs+2*num_stat_runs+i
        saveConfig(config=config, computername="experiment_7", trial_num=str(trial_num), folder_save=True)

