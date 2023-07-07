import sys; sys.path.append("/home/gonzaeve/boids/leader-follower")
import numpy as np
import matplotlib.pyplot as plt
from lib.file_helper import loadConfig, loadTrial

def getStatistics(trials, computer_name):
    save_datas = [loadTrial(trial, computer_name) for trial in trials]

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


def main():
    # Grab trials for each reward structure
    # trials_Dfollow = ["trial_999", "trial_1000", "trial_1001"]
    # trials_G = ["trial_1002", "trial_1003", "trial_1004"]
    # trials_D = ["trial_1005", "trial_1006", "trial_1007"]


    # trial_num = 1142
    # trial_num = 1569 # w. 20 trials shows D, Df are better than g
    # trial_num = 1665 # w. 20 trials shows G is better (this trial num might be wrong actually, did not track this well)
    # trial_num = 2229 # one the runs looking at varying coupling, num_stat_runs=20
    # trial_num = 2780 # w. 20 trials is where I start to trick D with followers
    # trial_num = 2598 # w 20 stat runs?? not sure what this trial number was for
    trial_num = 308
    num_stat_runs = 3
    computer_name = "graf200-16"

    tested_G = True
    tested_D = True
    tested_Dfollow = True
    tested_Zero = False
    plot_min_max_range = True

    start_trial_num = trial_num

    if tested_Zero:
        trials_Zero = []
        for i in range(num_stat_runs):
            trials_Zero.append("trial_"+str(trial_num))
            trial_num-=1
        print("Zero trials: ", trials_Zero)

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

    # def plotStatistics(middle, upper, lower):
    #     plt.figure(0)
    #     plt.plot()

    plt.figure(0)

    plt.ylim([0,1.01])

    # Get statistics for different reward structures
    legend = []
    if tested_G: 
        avg_G, std_dev_G, upper_dev_G, lower_dev_G, upper_range_G, lower_range_G = getStatistics(trials_G, computer_name)
        num_generations_arr = np.arange(avg_G.shape[0])
        plt.plot(num_generations_arr, avg_G, color='tab:blue')
        legend.append("$G$")

    if tested_D: 
        avg_D, std_dev_D, upper_dev_D, lower_dev_D, upper_range_D, lower_range_D = getStatistics(trials_D, computer_name)
        num_generations_arr = np.arange(avg_D.shape[0])
        plt.plot(num_generations_arr, avg_D, color='tab:orange')
        legend.append("$D$")

    if tested_Dfollow: 
        avg_Df, std_dev_Df, upper_dev_Df, lower_dev_Df, upper_range_Df, lower_range_Df = getStatistics(trials_Dfollow, computer_name)
        num_generations_arr = np.arange(avg_Df.shape[0])
        plt.plot(num_generations_arr, avg_Df, color="tab:green")
        legend.append(r'$D_{follow}$')

    if tested_Zero:
        avg_Z, std_dev_Z, upper_dev_Z, lower_dev_Z, upper_range_Z, lower_range_Z = getStatistics(trials_Zero, computer_name)
        num_generations_arr = np.arange(avg_Z.shape[0])
        plt.plot(num_generations_arr, avg_Z, color="tab:pink")
        legend.append("$Zero$")

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
    
    if tested_Zero:
        plt.fill_between(num_generations_arr, upper_dev_Z, lower_dev_Z, alpha=0.2, color="tab:pink")
        if plot_min_max_range:
            plt.fill_between(num_generations_arr, upper_range_Z, lower_range_Z, alpha=0.2, color="tab:pink")

    plt.legend(legend)

    # plt.legend(["$G$", "$D$", r'$D_{follow}$'])

    plt.xlabel("Number of Generations")
    plt.ylabel("Average Team Fitness")
    # plt.title("Reward Shaping with Informative G")

    # plt.xlim([0,150])

    plot_save_name = "figures/trail_"+str(start_trial_num)+" | stat_runs "+str(num_stat_runs)+" |"
    if tested_G:
        plot_save_name += " G"
    if tested_D:
        plot_save_name += " D"
    if tested_Dfollow:
        plot_save_name += " Df"
    if tested_Zero:
        plot_save_name += " Z"
    if plot_min_max_range:
        plot_save_name += " | full range"
    else:
        plot_save_name += " | std err"
    plot_save_name += " | " + computer_name
    plot_save_name += ".png"

    print("Saving plot as ", plot_save_name)
    plt.savefig(plot_save_name)

    plt.show()

if __name__ == "__main__":
    main()
