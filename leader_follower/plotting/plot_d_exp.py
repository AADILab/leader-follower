# Configs 247 - 291. Sweep of different numbers of learners to followers
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from leader_follower import project_properties
from leader_follower.file_helper import load_config, load_trial


def process_experiment(exp_dir, start_trial, stat_runs):
    comps = ["G with Followers", r"D$_{swarm}$ with Followers", "G", "D"]
    markers = ['o', '^', 'p', 'x']
    num_comps = len(comps)

    # todo use exp_dir here to build path to configs and trials
    configs = ["config_" + str(i + start_trial) + ".yaml" for i in range(num_comps * stat_runs)]
    trials = ["trial_" + str(i + start_trial) for i in range(num_comps * stat_runs)]

    # Get the average best performance of each one
    # todo use project_properties
    save_datas = [load_trial(trial) for trial in trials]
    loaded_configs = [load_config(config) for config in configs]

    # Plot the team performances
    data_dict = {}
    for n_var, each_comp in enumerate(comps):
        # This will hold average team fitnesses from CCEA population.
        # For each trial, we have the average team fitness for each generation
        all_team_fitnesses = []
        data_dict[each_comp] = {}
        # print(var, save_datas[n_var::stat_runs])
        for save_data, loaded_config, config in zip(
                save_datas[n_var::num_comps],
                loaded_configs[n_var::num_comps],
                configs[n_var::num_comps]
        ):
            # Grab the average fitness over time
            unfiltered_scores_list = save_data["unfiltered_scores_list"]
            print("Config: ", config, " | Nominal Variable:", each_comp, " | Diff Eval: ",
                  loaded_config["CCEA"]["use_difference_evaluations"], " | Num Leaders: ",
                  loaded_config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"], " | Score: ",
                  unfiltered_scores_list[-1])
            # Save that to all team_fitnesses
            all_team_fitnesses.append(unfiltered_scores_list)
        # Turn team fitnesses into an array
        all_team_fitness_arr = np.array(all_team_fitnesses)
        # Save that to data dictionary
        data_dict[each_comp]["all_team_fitness_arr"] = all_team_fitness_arr
        # print(np.average(all_team_fitness_arr, axis=0), '\n', np.average(all_team_fitness_arr, axis=0))
        # Save the average team fitness across trials
        data_dict[each_comp]["avg_team_fitness_arr"] = np.average(all_team_fitness_arr, axis=0)
        # Save the standard deviation across trials
        data_dict[each_comp]["std_dev_team_fitness_arr"] = np.std(all_team_fitness_arr, axis=0)
        # Save the upper std deviation
        data_dict[each_comp]["upper_std_dev_team_fitness_arr"] = data_dict[each_comp]["avg_team_fitness_arr"] + \
            data_dict[each_comp]["std_dev_team_fitness_arr"]
        # Save the lower std deviation
        data_dict[each_comp]["lower_std_dev_team_fitness_arr"] = data_dict[each_comp]["avg_team_fitness_arr"] - \
            data_dict[each_comp]["std_dev_team_fitness_arr"]
        # Save the upper range
        data_dict[each_comp]["upper_range"] = np.max(data_dict[each_comp]["avg_team_fitness_arr"], axis=0)
        # Save the lower range
        data_dict[each_comp]["lower_range"] = np.min(data_dict[each_comp]["avg_team_fitness_arr"], axis=0)

    # Plot data
    # # todo use project_properties
    num_generations_arr = np.arange(load_config(configs[0])["num_generations"] + 1)

    plt.figure(0)
    plt.ylim([-0.1, 1.0])
    print(comps)
    for each_comp, m in zip(comps, markers):
        plt.plot(num_generations_arr, data_dict[each_comp]["avg_team_fitness_arr"], marker=m)
    plt.legend(comps)
    for each_comp in comps:
        plt.fill_between(num_generations_arr, data_dict[each_comp]["upper_std_dev_team_fitness_arr"],
                         data_dict[each_comp]["lower_std_dev_team_fitness_arr"], alpha=0.2)
    # for var in vars:
    # plt.fill_between(num_generations_arr, data_dict[var]["upper_range"], data_dict[var]["lower_range"], alpha=0.2)
    plt.title("Different Learning Methods for AUV Observation Domain")
    plt.xlabel("Number of Generations")
    plt.ylabel("Average Team Fitness Score")
    plt.tight_layout()
    plt.show()
    return


def main():
    # Data Wrangling
    start_trial = 0
    stat_runs = 10

    exp_types = [
        'global',
        'difference_leader',
        'difference_leader_follower',
        'difference_proximity'
    ]
    base_dirs = [
        Path(project_properties.data_dir, each_exp)
        for each_exp in exp_types
    ]

    for exp_dir in base_dirs:
        process_experiment(exp_dir)
    return
