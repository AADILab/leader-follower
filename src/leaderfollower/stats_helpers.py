from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from leaderfollower.file_helper import loadConfig, loadTrial

def getStatistics(
        trials: List[str], 
        computer_name: str
    ):
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

    stats_dict = {
        "avg_team_fitness_arr": avg_team_fitness_arr, 
        "std_dev_team_fitness_arr": std_dev_team_fitness_arr, 
        "upper_err_team_fitness_arr": upper_err_team_fitness_arr, 
        "lower_err_team_fitness_arr": lower_err_team_fitness_arr, 
        "upper_range": upper_range, 
        "lower_range": lower_range
    }

    return stats_dict


def getBatchStatistics(
        last_trial_num: int, 
        num_stat_runs: int, 
        computer_name: str,
        tested_Zero: bool,
        tested_Dfollow: bool, 
        tested_D: bool,
        tested_G: bool,
    ):
    """ This gets all the statistics for a single batch of trials
    A batch is single combination of parameters run for a certain number of
    stat runs with the reward shaping methods specified. 
    
    You have to check manually that the methods tested are the same ones you
    put as tested for this function because there is no automatic check
    """
    trial_num = last_trial_num

    # Aggregate all of the trials by name
    if tested_Zero:
        trials_Zero = []
        for _ in range(num_stat_runs):
            trials_Zero.append("trial_"+str(trial_num))
            trial_num-=1
        print(f"Zero trials: {trials_Zero}")

    if tested_Dfollow:
        trials_Dfollow = []
        for _ in range(num_stat_runs):
            trials_Dfollow.append("trial_"+str(trial_num))
            trial_num -= 1
        print(f"Dfollow trials: {trials_Dfollow}")

    if tested_D:
        trials_D = []
        for _ in range(num_stat_runs):
            trials_D.append("trial_"+str(trial_num))
            trial_num -= 1
        print(f"D trials: {trials_D}")

    if tested_G:
        trials_G = []
        for _ in range(num_stat_runs):
            trials_G.append("trial_"+str(trial_num))
            trial_num -= 1
        print(f"G trials: {trials_G}")
    
    # Get the statistics for different reward shaping methods
    if tested_G: 
        stats_dict_G = getStatistics(trials_G, computer_name)
    else:
        stats_dict_G = "Not tested"

    if tested_D: 
        stats_dict_D = getStatistics(trials_D, computer_name)
    else:
        stats_dict_D = "Not tested"

    if tested_Dfollow: 
        stats_dict_Df = getStatistics(trials_Dfollow, computer_name)
    else:
        stats_dict_Df = "Not tested"

    if tested_Zero:
        stats_dict_Z = getStatistics(trials_Zero, computer_name)
    else:
        stats_dict_Z = "Not tested"
    
    return stats_dict_G, stats_dict_D, stats_dict_Df, stats_dict_Z

def get1DSweepStatistics(
        num_batches: int,
        last_trial_num: int, 
        num_stat_runs: int, 
        computer_name: str,
        tested_Zero: bool,
        tested_Dfollow: bool, 
        tested_D: bool,
        tested_G: bool
    ):
    """Aggregate all of the statistics accross all of the batches for a particular sweep of one parameter
    Just return the final value for each batch for average performance, standard deviation, etc

    num_batches is the number of variables that were tested for this particular parameter
    For example, for a coupling sweep of [1,3,5], num_batches would be 3
    """
    
    if tested_Zero:
        sweep_stats_Z = {
            "avg_team_fitness_arr": [], 
            "std_dev_team_fitness_arr": [], 
            "upper_err_team_fitness_arr": [], 
            "lower_err_team_fitness_arr": [], 
            "upper_range": [], 
            "lower_range": []
        }
    else:
        sweep_stats_Z = "Not tested"

    if tested_Dfollow:
        sweep_stats_Df = {
            "avg_team_fitness_arr": [], 
            "std_dev_team_fitness_arr": [], 
            "upper_err_team_fitness_arr": [], 
            "lower_err_team_fitness_arr": [], 
            "upper_range": [], 
            "lower_range": []
        }
    else:
        sweep_stats_Df = "Not tested"
    
    if tested_D:
        sweep_stats_D = {
            "avg_team_fitness_arr": [], 
            "std_dev_team_fitness_arr": [], 
            "upper_err_team_fitness_arr": [], 
            "lower_err_team_fitness_arr": [], 
            "upper_range": [], 
            "lower_range": []
        }
    else:
        sweep_stats_D = "Not tested"
    
    if tested_G:
        sweep_stats_G = {
            "avg_team_fitness_arr": [], 
            "std_dev_team_fitness_arr": [], 
            "upper_err_team_fitness_arr": [], 
            "lower_err_team_fitness_arr": [], 
            "upper_range": [], 
            "lower_range": []
        }
    else:
        sweep_stats_G = "Not tested"

    for batch_id in range(num_batches):
        last_batch_trial_num = last_trial_num - (num_batches-batch_id-1)*(tested_Zero+tested_Dfollow+tested_D+tested_G)*num_stat_runs
        print(f"Starting batch {batch_id} at trial number {last_batch_trial_num}")
        # Get the stats for this batch
        stats_dict_G, stats_dict_D, stats_dict_Df, stats_dict_Z = getBatchStatistics(
            last_batch_trial_num,
            num_stat_runs, 
            computer_name,
            tested_Zero,
            tested_Dfollow,
            tested_D,
            tested_G
        )
        # Aggregate those stats into the sweep stats
        if tested_G:
            for key in stats_dict_G:
                sweep_stats_G[key].append(stats_dict_G[key][-1])
        
        if tested_D:
            for key in stats_dict_D:
                sweep_stats_D[key].append(stats_dict_D[key][-1])
        
        if tested_Dfollow:
            for key in stats_dict_Df:
                sweep_stats_Df[key].append(stats_dict_Df[key][-1])
        
        if tested_Zero:
            for key in stats_dict_Z:
                sweep_stats_Z[key].append(stats_dict_Z[key][-1])

    # Turn lists into numpy arrays
    if tested_G:
        for key in sweep_stats_G:
            sweep_stats_G[key] = np.array(sweep_stats_G[key])
    
    if tested_D:
        for key in sweep_stats_D:
            sweep_stats_D[key] = np.array(sweep_stats_D[key])
    
    if tested_Dfollow:
        for key in sweep_stats_Df:
            sweep_stats_Df[key] = np.array(sweep_stats_Df[key])
    
    if tested_Zero:
        for key in sweep_stats_Z:
            sweep_stats_Z[key] = np.array(sweep_stats_Z[key])

    return sweep_stats_G, sweep_stats_D, sweep_stats_Df, sweep_stats_Z
