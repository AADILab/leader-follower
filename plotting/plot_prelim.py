import sys; sys.path.append("/home/gonzaeve/boids/leader-follower")
import numpy as np
import matplotlib.pyplot as plt
from lib.file_helper import loadConfig, loadTrial

# Grab trials for each reward structure
# trials_Dfollow = ["trial_999", "trial_1000", "trial_1001"]
# trials_G = ["trial_1002", "trial_1003", "trial_1004"]
# trials_D = ["trial_1005", "trial_1006", "trial_1007"]


# trial_num = 1142
# trial_num = 1569 # w. 20 trials shows D, Df are better than g
# trial_num = 1665 # w. 20 trials shows G is better (this trial num might be wrong actually, did not track this well)
trial_num = 1989
num_stat_runs = 20

tested_G = True
tested_D = True
tested_Dfollow = True
plot_min_max_range = True

if tested_Dfollow:
    trials_Dfollow = []
    for i in range(num_stat_runs):
        trials_Dfollow.append("trial_"+str(trial_num))
        trial_num -= 1
    print("Dfollow trials: ", trials_Dfollow)

if tested_D:
    trials_D = []
    for i in range(num_stat_runs):
        trials_D.append("trial_"+str(trial_num))
        trial_num -= 1
    print("D trials: ", trials_D)

if tested_G:
    trials_G = []
    for i in range(num_stat_runs):
        trials_G.append("trial_"+str(trial_num))
        trial_num -= 1
    print("G trials: ", trials_G)

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
    upper_err_team_fitness_arr = avg_team_fitness_arr + std_dev_team_fitness_arr/np.sqrt(all_team_fitness_arr.shape[0])
    lower_err_team_fitness_arr = avg_team_fitness_arr - std_dev_team_fitness_arr/np.sqrt(all_team_fitness_arr.shape[0])
    upper_range = np.max(all_team_fitness_arr, axis=0)
    lower_range = np.min(all_team_fitness_arr, axis=0)

    return avg_team_fitness_arr, std_dev_team_fitness_arr, upper_err_team_fitness_arr, lower_err_team_fitness_arr, upper_range, lower_range

# def plotStatistics(middle, upper, lower):
#     plt.figure(0)
#     plt.plot()

plt.figure(0)

plt.ylim([0,1.0])

# Get statistics for different reward structures
legend = []
if tested_G: 
    avg_G, std_dev_G, upper_dev_G, lower_dev_G, upper_range_G, lower_range_G = getStatistics(trials_G)
    num_generations_arr = np.arange(avg_G.shape[0])
    plt.plot(num_generations_arr, avg_G, color='tab:blue')
    legend.append("$G$")

if tested_D: 
    avg_D, std_dev_D, upper_dev_D, lower_dev_D, upper_range_D, lower_range_D = getStatistics(trials_D)
    num_generations_arr = np.arange(avg_D.shape[0])
    plt.plot(num_generations_arr, avg_D, color='tab:orange')
    legend.append("$D$")

if tested_Dfollow: 
    avg_Df, std_dev_Df, upper_dev_Df, lower_dev_Df, upper_range_Df, lower_range_Df = getStatistics(trials_Dfollow)
    num_generations_arr = np.arange(avg_Df.shape[0])
    plt.plot(num_generations_arr, avg_Df, color="tab:green")
    legend.append(r'$D_{follow}$')

if tested_G: 
    plt.fill_between(num_generations_arr, upper_dev_G, lower_dev_G, alpha=0.2, color="tab:blue")
    if plot_min_max_range:
        plt.fill_between(num_generations_arr, upper_range_G, lower_range_G, alpha=0.2, color="tab:blue")

if tested_D: 
    plt.fill_between(num_generations_arr, upper_dev_D, lower_dev_D, alpha=0.2, color="tab:orange")
    if plot_min_max_range:
        plt.fill_between(num_generations_arr, upper_range_D, lower_range_D, alpha=0.2, color="tab:orange")

if tested_Dfollow: 
    plt.fill_between(num_generations_arr, upper_dev_Df, lower_dev_Df, alpha=0.2, color="tab:green")
    if plot_min_max_range:
        plt.fill_between(num_generations_arr, upper_range_Df, lower_range_Df, alpha=0.2, color="tab:green")

plt.legend(legend)

# plt.legend(["$G$", "$D$", r'$D_{follow}$'])

plt.xlabel("Number of Generations")
plt.ylabel("Average Team Fitness")
# plt.title("Reward Shaping with Informative G")

# plt.xlim([0,150])

plt.show()
