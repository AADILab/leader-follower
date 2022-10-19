# Configs 247 - 291. Sweep of different numbers of learners to followers

import enum
import sys; sys.path.append("/home/egonzalez/leaders")
from sys import exit
import numpy as np
import matplotlib.pyplot as plt
from lib.file_helper import loadConfig, loadTrial
from copy import deepcopy

######## Data Wrangling

start_trial = 783
vars = ["G with Followers", r"D$_{swarm}$ with Followers", "G", "D"]
markers = ['o', '^', 'p', 'x']
num_var = len(vars)
stat_runs = 10

configs = ["config_"+str(i+start_trial)+".yaml" for i in range(num_var*stat_runs)]
trials = ["trial_"+str(i+start_trial) for i in range(num_var*stat_runs)]

# Get the average best performance of each one
save_datas = [loadTrial(trial) for trial in trials]
loaded_configs = [loadConfig(config) for config in configs]

######### Plot the team performances
data_dict = {}
for n_var, var in enumerate(vars):
    # This will hold average team fitnesses from CCEA population.
    # For each trial, we have the average team fitness for each generation
    all_team_fitnesses = []
    data_dict[var] = {}
    # print(var, save_datas[n_var::stat_runs])
    for save_data, loaded_config, config in zip(save_datas[n_var::num_var], loaded_configs[n_var::num_var], configs[n_var::num_var]):
        # Grab the average fitness over time
        unfiltered_scores_list = save_data["unfiltered_scores_list"]
        print("Config: ", config ," | Nominal Variable:", var, " | Diff Eval: ", loaded_config["CCEA"]["use_difference_evaluations"], " | Num Leaders: ", loaded_config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"]," | Score: ", unfiltered_scores_list[-1])
        # Save that to all team_fitnesses
        all_team_fitnesses.append(unfiltered_scores_list)
    # Turn team fitnesses into an array
    all_team_fitness_arr = np.array(all_team_fitnesses)
    # Save that to data dictionary
    data_dict[var]["all_team_fitness_arr"] = all_team_fitness_arr
    # print(np.average(all_team_fitness_arr, axis=0), '\n', np.average(all_team_fitness_arr, axis=0))
    # Save the average team fitness accross trials
    data_dict[var]["avg_team_fitness_arr"] = np.average(all_team_fitness_arr, axis=0)
    # Save the standard deviation accross trials
    data_dict[var]["std_dev_team_fitness_arr"] = np.std(all_team_fitness_arr, axis=0)
    # Save the upper std deviation
    data_dict[var]["upper_std_dev_team_fitness_arr"] = data_dict[var]["avg_team_fitness_arr"] + data_dict[var]["std_dev_team_fitness_arr"]
    # Save the lower std deviation
    data_dict[var]["lower_std_dev_team_fitness_arr"] = data_dict[var]["avg_team_fitness_arr"] - data_dict[var]["std_dev_team_fitness_arr"]
    # Save the upper range
    data_dict[var]["upper_range"] = np.max(data_dict[var]["avg_team_fitness_arr"], axis=0)
    # Save the lower range
    data_dict[var]["lower_range"] = np.min(data_dict[var]["avg_team_fitness_arr"], axis=0)

######## Plot data

num_generations_arr = np.arange(loadConfig(configs[0])["num_generations"]+1)

plt.figure(0)
plt.ylim([-0.1,1.0])
print(vars)
for var, m in zip(vars, markers):
    plt.plot(num_generations_arr, data_dict[var]["avg_team_fitness_arr"], marker=m)
plt.legend(vars)
for var in vars:
    plt.fill_between(num_generations_arr, data_dict[var]["upper_std_dev_team_fitness_arr"], data_dict[var]["lower_std_dev_team_fitness_arr"], alpha=0.2)
# for var in vars:
    # plt.fill_between(num_generations_arr, data_dict[var]["upper_range"], data_dict[var]["lower_range"], alpha=0.2)
plt.title("Different Learning Methods for AUV Observation Domain")
plt.xlabel("Number of Generations")
plt.ylabel("Average Team Fitness Score")
plt.tight_layout()
plt.show()

######### This was an attempt to plot all agent fitnesses accross all trials in one plot
#           This doesn't actually make much sense for a few reasons
#           1. Agent n in one trial is not necessarily going to have the same role as agent n in another trial, meaning it doesn't make sense to compare them accross those trials
#           2. All we really care about is the team performance overall, not individual agents. Individual agent performances are important for diagnosing why we see certain teams perform better than others, but that isn't the punchline
#           3. There are different numbers of learning agents in different trials. Specifically when we have a fixed team size, and we vary how many agents on that team are leaders vs followers

# data_dict = {}
# for n_var, var in enumerate(vars):
#     all_avg_fitnesses = []
#     for save_data in save_datas[n_var::stat_runs]:
#         # Grab the average fitnesses over time
#         average_agent_fitness_lists_unfiltered = save_data["average_agent_fitness_lists_unfiltered"]
#         # Save that to avg fitnesses, which holds fitnesses for all of this stat run
#         all_avg_fitnesses.append(average_agent_fitness_lists_unfiltered)
#     data_dict[var] = {}
#     print(data_dict)
#     print(data_dict[var])
#     print(all_avg_fitnesses)
#     print(len(all_avg_fitnesses))
#     print([len(e) for e in all_avg_fitnesses])
#     print([ [len(a) for a in e] for e in all_avg_fitnesses])
#     print(len(all_avg_fitnesses[0]))
#     print(len(all_avg_fitnesses[0][0]))
#     import sys; sys.exit()
#     np.array(all_avg_fitnesses)
#     data_dict[var]["all_avg_fitnesses"] = deepcopy(np.array(all_avg_fitnesses, dtype=float))
#     data_dict[var]["avg_accross_replicates"] = np.average(data_dict[var]["all_avg_fitnesses"], axis=0)
#     data_dict[var]["std_dev_accross_replicates"] = np.std(data_dict[var]["all_avg_fitnesses"], axis=0)
#     data_dict[var]["upper_std_dev_accross_replicates"] = data_dict[var]["avg_accross_replicates"] + data_dict[var]["std_dev_accross_replicates"]
#     data_dict[var]["lower_std_dev_accross_replicates"] = data_dict[var]["avg_accross_replicates"] - data_dict[var]["std_dev_accross_replicates"]

# num_generations_arr = np.arange(configs[0]["num_generations"]+1)

# plt.figure(0)

# plt.ylim([0.0,1.0])
# for var in vars:
#     plt.plot(num_generations_arr, data_dict[var]["avg_accross_replicates"])

# plt.show()
# plt.xlim()

# best_fitnesses = [save_data["best_team_data"].fitness for save_data in save_datas]
# averages = [np.average(best_fitnesses[(stat_runs*i):(stat_runs*i)+stat_runs]) for i in range(num_var)]
# uppers = [averages[i]+np.std(best_fitnesses[(stat_runs*i):(stat_runs*i)+stat_runs]) for i in range(num_var)]
# lowers = [averages[i]-np.std(best_fitnesses[(stat_runs*i):(stat_runs*i)+stat_runs]) for i in range(num_var)]

# # Get the recorded ratio of learners to followers
# config_dicts = [loadConfig(config) for config in configs]
# num_followers = [config_dict["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"]  for config_dict in config_dicts][::stat_runs]
# # ratios = [(15.-num_follow)/num_follow for num_follow in num_followers]



# plt.figure(0)
# plt.ylim([0.0, 1.0])

# plt.fill_between(x=num_followers,y1=lowers, y2=uppers, alpha=0.2)
# plt.plot(num_followers, averages)
# plt.xticks(num_followers)
# plt.ylabel("System Performance")
# plt.xlabel("Number of Followers")
# plt.title("Varying Number of Followers out of 15 agents.")
# plt.show()
