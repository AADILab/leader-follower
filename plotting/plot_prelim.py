import sys; sys.path.append("/home/gonzaeve/boids/leader-follower")
import numpy as np
import matplotlib.pyplot as plt
from lib.file_helper import loadConfig, loadTrial

# Grab trials for each reward structure
trials_Dfollow = ["trial_999", "trial_1000", "trial_1001"]
trials_G = ["trial_1002", "trial_1003", "trial_1004"]
trials_D = ["trial_1005", "trial_1006", "trial_1007"]

def getStatistics(trials):
    save_datas = [loadTrial(trial) for trial in trials]

    # Grab the fitnesses accross the trials
    all_team_fitnesses = []
    for save_data in save_datas:
        unfiltered_scores_list = save_data["unfiltered_scores_list"]
        all_team_fitnesses.append(unfiltered_scores_list)

    # Turn team fitnesses into an array
    all_team_fitness_arr = np.array(all_team_fitnesses)

    # Get statistics accross these runs
    avg_team_fitness_arr = np.average(all_team_fitness_arr, axis=0)
    std_dev_team_fitness_arr = np.std(all_team_fitness_arr, axis=0)
    upper_std_dev_team_fitness_arr = avg_team_fitness_arr + std_dev_team_fitness_arr
    lower_std_dev_team_fitness_arr = avg_team_fitness_arr - std_dev_team_fitness_arr
    upper_range = np.max(avg_team_fitness_arr, axis=0)
    lower_range = np.min(avg_team_fitness_arr, axis=0)

    return avg_team_fitness_arr, std_dev_team_fitness_arr, upper_std_dev_team_fitness_arr, lower_std_dev_team_fitness_arr, upper_range, lower_range

# def plotStatistics(middle, upper, lower):
#     plt.figure(0)
#     plt.plot()

# Get statistics for different reward structures
avg_G, _, upper_dev_G, lower_dev_G, _, _ = getStatistics(trials_G)
avg_D, _, upper_dev_D, lower_dev_D, _, _ = getStatistics(trials_D)
avg_Df, _, upper_dev_Df, lower_dev_Df, _, _ = getStatistics(trials_Dfollow)

num_generations_arr = np.arange(avg_G.shape[0])

plt.figure(0)

plt.ylim([0,16])

plt.plot(num_generations_arr, avg_G)
plt.plot(num_generations_arr, avg_D)
plt.plot(num_generations_arr, avg_Df)

plt.fill_between(num_generations_arr, upper_dev_G, lower_dev_G, alpha=0.2)
plt.fill_between(num_generations_arr, upper_dev_D, lower_dev_D, alpha=0.2)
plt.fill_between(num_generations_arr, upper_dev_Df, lower_dev_Df, alpha=0.2)

plt.legend(["$G$", "$D$", r'$D_{follow}$'])

plt.xlabel("Number of Generations")
plt.ylabel("Average Team Fitness")
plt.title("Reward Shaping with Informative G")

plt.show()
