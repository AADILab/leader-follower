""" These functions are for parsing the data from loadTrialData() into more easily manageable pieces of data""" 

import numpy as np
import os
from typing import List

def getEvalFitnesses(trial_data: dict):
    # List of floating point values. Each value is a team fitness
    team_fitnesses = []
    # List of sublists of floating point values. Each sublist is the agent fitnesses at that generation
    agent_fitnesses = []

    for generation_data in trial_data:
        team_fitnesses.append(generation_data["evaluation_team"]["team_fitness"])
        agent_fitnesses.append(generation_data["evaluation_team"]["agent_fitnesses"])
    
    return np.array(team_fitnesses), np.array(agent_fitnesses)

def getAllEvalFitnesses(trial_datas: List[dict]):
    # Get all the team fitness and agent specific fitnesses for the evaluation teams
    all_team_fitnesses = []
    all_agent_fitnesses = []
    for trial_data in trial_datas:
        team_fitnesses, agent_fitnesses = getEvalFitnesses(trial_data)
        all_team_fitnesses.append(team_fitnesses)
        all_agent_fitnesses.append(agent_fitnesses)
    return np.array(all_team_fitnesses), np.array(all_agent_fitnesses)

def getEvalStatistics(trial_datas: List[dict]):
    # This assumes that all of these trials come from the same config
    # (Each trial was run with the same exact configuration)

    # Get all the team fitnesses and agent fitnesses out
    all_team_fitnesses, _ = getAllEvalFitnesses(trial_datas)

    # Get statistics accross these runs
    avg_team_fitness_arr = np.average(all_team_fitnesses, axis=0)
    std_dev_team_fitness_arr = np.std(all_team_fitnesses, axis=0)
    upper_err_team_fitness_arr = avg_team_fitness_arr + std_dev_team_fitness_arr/np.sqrt(all_team_fitnesses.shape[0])
    lower_err_team_fitness_arr = avg_team_fitness_arr - std_dev_team_fitness_arr/np.sqrt(all_team_fitnesses.shape[0])
    upper_range = np.max(all_team_fitnesses, axis=0)
    lower_range = np.min(all_team_fitnesses, axis=0)

    return avg_team_fitness_arr, std_dev_team_fitness_arr, upper_err_team_fitness_arr, lower_err_team_fitness_arr, upper_range, lower_range

def getBestFitnesses(trial_data: dict):
    # List of floating point values. Each value is a team fitness
    team_fitnesses = []
    # List of sublists of floating point values. Each sublist is the agent fitnesses at that generation
    # We use the highest score of an agent across its entire population. We dont' make this dependent on whether this particular
    # policy was used in the random team that got the best team fitness
    agent_fitnesses = []

    for generation_data in trial_data:
        team_fitnesses = [generation_data["training_teams"][team_name]["team_fitness"] for team_name in generation_data["training_teams"].keys()]
        best_team_fitness = max(team_fitnesses)

        agent_fitnesses = [generation_data["training_teams"][team_name]["agent_fitnesses"] for team_name in generation_data["training_teams"].keys()]
        best_agent_fitnesses = [max(agent_fitness_list) for agent_fitness_list in agent_fitnesses]

        team_fitnesses.append(best_team_fitness)
        agent_fitnesses.append(best_agent_fitnesses)

    return team_fitnesses, agent_fitnesses

def getAllBestFitnesses(trial_datas: List[dict]):
    # Get all the team fitness and agent specific best fitnesses for the training teams
    all_team_fitnesses = []
    all_agent_fitnesses = []
    for trial_data in trial_datas:
        team_fitnesses, agent_fitnesses = getBestFitnesses(trial_data)
        all_team_fitnesses.append(team_fitnesses)
        all_agent_fitnesses.append(agent_fitnesses)
    return np.array(all_team_fitnesses), np.array(all_agent_fitnesses)

def getBestStatistics(trial_datas: List[dict]):
    # This assumes that all of these trials come from the same config
    # (Each trial was run with the same exact configuration)

    # Get all the team fitnesses out
    all_team_fitnesses, _ = getAllBestFitnesses(trial_datas)

    # Get statistics accross these runs
    avg_team_fitness_arr = np.average(all_team_fitnesses, axis=0)
    std_dev_team_fitness_arr = np.std(all_team_fitnesses, axis=0)
    upper_err_team_fitness_arr = avg_team_fitness_arr + std_dev_team_fitness_arr/np.sqrt(all_team_fitnesses.shape[0])
    lower_err_team_fitness_arr = avg_team_fitness_arr - std_dev_team_fitness_arr/np.sqrt(all_team_fitnesses.shape[0])
    upper_range = np.max(all_team_fitnesses, axis=0)
    lower_range = np.min(all_team_fitnesses, axis=0)

    return avg_team_fitness_arr, std_dev_team_fitness_arr, upper_err_team_fitness_arr, lower_err_team_fitness_arr, upper_range, lower_range

def getTrialNames(trial_num: int, num_stat_runs: int):
    trialnames = []
    for i in range(num_stat_runs):
        trialnames.append("trial_"+str(trial_num))
        trial_num -= 1
    return trialnames, trial_num

