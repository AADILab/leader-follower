import sys; sys.path.append("/home/viswansi/for_merging/leader-follower")
from sys import exit
import numpy as np
import matplotlib.pyplot as plt
from lib.file_helper import loadConfig, loadTrial
from tqdm import tqdm
from copy import deepcopy

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

def getFinalEvalStatistics(trials, computer_name):
    # save_datas = []
    all_final_evaluation_teams_fitnesses = []
    for trial in tqdm(trials):
        # Grab the save data for this trial
        save_data = loadTrial(trial, computer_name)
        # save_datas.append(loadTrial(trial, computer_name))

        # Grab the piece of this data we care about
        final_evaluation_teams_fitnesses = deepcopy([team_data.fitness for team_data in save_data["final_evaluation_teams"]])

        # Append that to our list
        all_final_evaluation_teams_fitnesses.append(final_evaluation_teams_fitnesses)
        
        del save_data


        # save_datas[-1].close()
    # all_final_evaluation_teams_fitnesses = []

    # for save_data in tqdm(save_datas):
    #     final_evaluation_teams_fitnesses = [team_data.fitness for team_data in save_data["final_evaluation_teams"]]
    #     all_final_evaluation_teams_fitnesses.append(final_evaluation_teams_fitnesses)
    
    all_final_evaluation_teams_fitnesses_arr = np.array(all_final_evaluation_teams_fitnesses)
    #all_final_evaluation_teams_fitnesses now has an array like this [[score_gen0, gen1, ...], [score_gen0, gen1, ...], ...]
    #calculate the average of the score at gen0, gen1, etc.

    avg_final_evaluation_teams_fitnesses_arr = np.average(all_final_evaluation_teams_fitnesses_arr, axis=0)
    std_dev_final_evaluation_teams_fitnesses_arr = np.std(all_final_evaluation_teams_fitnesses_arr, axis=0)
    upper_err_final_evaluation_teams_fitnesses_arr = avg_final_evaluation_teams_fitnesses_arr + std_dev_final_evaluation_teams_fitnesses_arr/np.sqrt(all_final_evaluation_teams_fitnesses_arr.shape[0])
    lower_err_final_evaluation_teams_fitnesses_arr = avg_final_evaluation_teams_fitnesses_arr - std_dev_final_evaluation_teams_fitnesses_arr/np.sqrt(all_final_evaluation_teams_fitnesses_arr.shape[0])
    upper_range = np.max(all_final_evaluation_teams_fitnesses_arr, axis=0)
    lower_range = np.min(all_final_evaluation_teams_fitnesses_arr, axis=0)

    return avg_final_evaluation_teams_fitnesses_arr, std_dev_final_evaluation_teams_fitnesses_arr, upper_err_final_evaluation_teams_fitnesses_arr, lower_err_final_evaluation_teams_fitnesses_arr, upper_range, lower_range


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
    trial_num = 79
    num_stat_runs = 20
    computer_name = "drip_2b_experiments"

    tested_FCoupleD = False
    tested_FCouple = False
    tested_G = True
    tested_Dfollow = True
    tested_drip_FCoupleD = True
    tested_drip_FCouple = True
    
    plot_min_max_range = False

    start_trial_num = trial_num

    if tested_drip_FCoupleD:
        trials_drip_FCoupleD = []
        for i in range(num_stat_runs):
            trials_drip_FCoupleD.append("trial_"+str(trial_num))
            trial_num-=1
        print("DRiP FCoupleD trials: ", trials_drip_FCoupleD)
        print()
    if tested_drip_FCouple:
        trials_drip_FCouple = []
        for i in range(num_stat_runs):
            trials_drip_FCouple.append("trial_"+str(trial_num))
            trial_num-=1
        print("DRiP FCouple trials: ", trials_drip_FCouple)
        print()
    
    if tested_Dfollow:
        trials_DFollow = []
        for i in range(num_stat_runs):
            trials_DFollow.append("trial_"+str(trial_num))
            trial_num-=1
        print("DFollow trials: ", trials_DFollow)
        print()

    if tested_G:
        trials_G = []
        for i in range(num_stat_runs):
            trials_G.append("trial_"+str(trial_num))
            trial_num-=1
        print("G trials: ", trials_G)
        print()
    

    if tested_FCouple:
        trials_FCouple = []
        for i in range(num_stat_runs):
            trials_FCouple.append("trial_"+str(trial_num))
            trial_num-=1
        print("FCouple trials: ", trials_FCouple)
        print()

    if tested_FCoupleD:
        trials_FCoupleD = []
        for i in range(num_stat_runs):
            trials_FCoupleD.append("trial_"+str(trial_num))
            trial_num-=1
        print("FCoupleD trials: ", trials_FCoupleD)
        print()
    
    
    
    
    
    #exit()

    # def plotStatistics(middle, upper, lower):
    #     plt.figure(0)
    #     plt.plot()

    plt.figure(0)

    plt.ylim([0,1.0])

    # Get statistics for different reward structures
    legend = []
    if tested_G: 
        avg_G, std_dev_G, upper_dev_G, lower_dev_G, upper_range_G, lower_range_G = getFinalEvalStatistics(trials_G, computer_name)
        num_generations_arr = np.arange(avg_G.shape[0])
        plt.plot(num_generations_arr, avg_G, color='tab:blue')
        legend.append("$G$")

    if tested_Dfollow: 
        avg_Df, std_dev_Df, upper_dev_Df, lower_dev_Df, upper_range_Df, lower_range_Df = getFinalEvalStatistics(trials_DFollow, computer_name)
        num_generations_arr = np.arange(avg_Df.shape[0])
        plt.plot(num_generations_arr, avg_Df, color="tab:green")
        legend.append(r'$D_{Indirect}$')

    if tested_FCouple: 
        avg_D, std_dev_D, upper_dev_D, lower_dev_D, upper_range_D, lower_range_D = getFinalEvalStatistics(trials_FCouple, computer_name)
        num_generations_arr = np.arange(avg_D.shape[0])
        plt.plot(num_generations_arr, avg_D, color='tab:olive')
        legend.append("$Gather$")

    if tested_FCoupleD: 
        avg_FCD, std_dev_FCD, upper_dev_FCD, lower_dev_FCD, upper_range_FCD, lower_range_FCD = getFinalEvalStatistics(trials_FCoupleD, computer_name)
        num_generations_arr = np.arange(avg_FCD.shape[0])
        plt.plot(num_generations_arr, avg_FCD, color='tab:orange')
        legend.append("$Steer + Gather$")
    
    if tested_drip_FCoupleD: 
        avg_Dr_FCD, std_dev_Dr_FCD, upper_dev_Dr_FCD, lower_dev_Dr_FCD, upper_range_Dr_FCD, lower_range_Dr_FCD = getFinalEvalStatistics(trials_drip_FCoupleD, computer_name)
        num_generations_arr = np.arange(avg_Dr_FCD.shape[0])
        plt.plot(num_generations_arr, avg_Dr_FCD, color="tab:red")
        legend.append(r'$DRiP_{Steer + Gather}$')

    if tested_drip_FCouple:
        avg_FCouple, std_dev_FCouple, upper_dev_FCouple, lower_dev_FCouple, upper_range_FCouple, lower_range_FCouple = getFinalEvalStatistics(trials_drip_FCouple, computer_name)
        num_generations_arr = np.arange(avg_FCouple.shape[0])
        plt.plot(num_generations_arr, avg_FCouple, color="tab:pink")
        legend.append(r"$DRiP_{Gather}$")

    # if tested_G: 
    #     plt.fill_between(num_generations_arr, upper_dev_G, lower_dev_G, alpha=0.2, color="tab:blue")
    #     if plot_min_max_range:
    #         plt.fill_between(num_generations_arr, upper_range_G, lower_range_G, alpha=0.2, color="tab:blue")

    # if tested_FCoupleD: 
    #     plt.fill_between(num_generations_arr, upper_dev_FCD, lower_dev_FCD, alpha=0.2, color="tab:orange")
    #     if plot_min_max_range:
    #         plt.fill_between(num_generations_arr, upper_range_FCD, lower_range_FCD, alpha=0.2, color="tab:orange")
    
    # if tested_FCouple: 
    #     plt.fill_between(num_generations_arr, upper_dev_D, lower_dev_D, alpha=0.2, color="tab:olive")
    #     if plot_min_max_range:
    #         plt.fill_between(num_generations_arr, upper_range_D, lower_range_D, alpha=0.2, color="tab:olive")

    # if tested_Dfollow: 
    #     plt.fill_between(num_generations_arr, upper_dev_Df, lower_dev_Df, alpha=0.2, color="tab:green")
    #     if plot_min_max_range:
    #         plt.fill_between(num_generations_arr, upper_range_Df, lower_range_Df, alpha=0.2, color="tab:green")
    
    # if tested_drip_FCoupleD:
    #     plt.fill_between(num_generations_arr, upper_dev_Dr_FCD, lower_dev_Dr_FCD, alpha=0.2, color="tab:red")
    #     if plot_min_max_range:
    #         plt.fill_between(num_generations_arr, upper_range_Dr_FCD, lower_range_Dr_FCD, alpha=0.2, color="tab:red")
    
    # if tested_drip_FCouple:
    #     plt.fill_between(num_generations_arr, upper_dev_FCouple, lower_dev_FCouple, alpha=0.2, color="tab:pink")
    #     if plot_min_max_range:
    #         plt.fill_between(num_generations_arr, upper_range_FCouple, lower_range_FCouple, alpha=0.2, color="tab:pink")
    

    plt.legend(legend)

    # plt.legend(["$G$", "$D$", r'$D_{follow}$'])

    plt.xlabel("Number of Generations")
    plt.ylabel("Average Team Fitness")
    # plt.title("Reward Shaping with Informative G")

    # plt.xlim([0,150])

    plot_save_name = "figures/driptest"
    if tested_G:
        plot_save_name += " G"
    if tested_FCoupleD:
        plot_save_name += " FCoupleD"
    if tested_Dfollow:
        plot_save_name += " Df"
    if tested_drip_FCouple:
        plot_save_name += " DRiP_FC"
    if plot_min_max_range:
        plot_save_name += " | full range"
    else:
        plot_save_name += " | std err"
    plot_save_name += " | " + computer_name
    plot_save_name += ".png"

    plot_save_name = "figures/drip_finaleval_2b_masters.png"

    print("Saving plot as ", plot_save_name)
    plt.savefig(plot_save_name)

    # plt.show()

if __name__ == "__main__":
    main()
