""" These functions are for parsing the data from loadTrialData() into more easily manageable pieces of data""" 

import numpy as np

def getEvalFitnesses(trial_data: dict):
    # List of floating point values. Each value is a team fitness
    team_fitnesses = []
    # List of sublists of floating point values. Each sublist is the agent fitnesses at that generation
    agent_fitnesses = []

    for generation_data in trial_data:
        team_fitnesses.append(generation_data["evaluation_team"]["team_fitness"])
        agent_fitnesses.append(generation_data["agent_fitness"])
    
    return np.array(team_fitnesses), np.array(agent_fitnesses)

# def getBestTrainFitnesses(trial_data: dict):
#     # List of floating point values. Each value is a team fitness
#     team_fitnesses = []
#     # List of sublists of floating point values. Each sublist is the agent fitnesses at that generation
#     agent_fitnesses = []

#     for generation_data in trial_data:
        