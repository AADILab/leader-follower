from typing import Dict, List, Optional
import pickle
from lib.network_lib import NN
from os import listdir, makedirs
from os.path import isfile, join, exists
import yaml
import myaml
import socket
import numpy as np

def getHostName():
    return socket.gethostname()

def loadTrial(trialname: str, computername: Optional[str]) -> Dict:
    # By default load the trial from this computer's save directory
    if computername is None:
        computername = getHostName()
    trial_path = join("results", computername, "trials", trialname+".pkl")
    return pickle.load(open(trial_path, "rb"))

def generateTeamDict(team_dir: str):
    fitness_npz = join(team_dir, "fitnesses.npz")
    traj_npz = join(team_dir, "joint_trajectory.npz")

    # Load in the team data
    team_fitnesses = np.load(fitness_npz)
    team_fitness = team_fitnesses["team_fitness"][0]
    agent_fitness = team_fitnesses["agent_fitness"]
    policy_ids = team_fitnesses["policy_ids"]

    traj_loaded = np.load(traj_npz)
    traj = traj_loaded["joint_trajectory"]

    team_dict = {
        "team_fitness": team_fitness,       # floating point value for team fitness
        "agent_fitness": agent_fitness,     # 1D np array for agent specific fitnesses
        "policy_ids": policy_ids,           # 1D np array of int values for which policy each agent used from its population
        "joint_trajectory": traj            # 3D np array representing joint trajectory that team took
        # joint_trajectory | 1st dimension is step in time. 2D slice represents [xs, ys] of all leaders and followers at this point in time
    }

    return team_dict

def loadTrialData(trialname: str, computername: Optional[str]):
    if computername is None:
        computername = getHostName()
    
    trial_path = join("results", computername, "trials", trialname)

    # Set up generations list so we can traverse them
    files_list = listdir(path=trial_path)
    generations_list = [file for file in files_list if file[:11]=="generation_"]
    generations_numbers = sorted([int(generation_name.split("_")[-1]) for generation_name in generations_list])
    generation_folders = ["generation_"+str(number) for number in generations_numbers]

    # Go through each generation and store relevant information
    trial_data = []
    for generation_name in generation_folders:
        generation_dir = join(trial_path, generation_name)

        evaluation_team_dir = join(generation_dir, "evaluation_team")
        
        population_dir = join(generation_dir, "population")
        leader_names = listdir(population_dir)
        leader_pop_dirs = [join(population_dir, leader_name) for leader_name in leader_names]

        generation_data = {
            "evaluation_team": generateTeamDict(evaluation_team_dir),
            "population": {
                leader_name: [] for leader_name in leader_names
            },
            "training_teams": {}
        }

        # Add in the population information
        for leader_pop_dir, leader_name in zip(leader_pop_dirs, leader_names):
            policy_names = listdir(path=leader_pop_dir)
            policy_numbers = sorted([int(policy_name.split("_")[-1][:-4]) for policy_name in policy_names])
            policy_npz_files = ["policy_"+str(number)+".npz" for number in policy_numbers]
            policy_npz_dirs = [join(leader_pop_dir, policy_npz) for policy_npz in policy_npz_files]

            for policy_npz in policy_npz_dirs:
                generation_data["population"][leader_name].append(np.load(policy_npz))
        
        # Add in training teams information
        training_teams_dir = join(generation_dir, "training_teams")
        team_names = listdir(path=training_teams_dir)
        for team_name in team_names:
            train_team_dir = join(training_teams_dir, team_name)
            generation_data["training_teams"][team_name] = generateTeamDict(train_team_dir)

        # Add all this generation data to the trial data
        trial_data.append(generation_data)
    
    return trial_data


# ,
#             "population": {
#                 leader_name: [np.load()] for (leader_name, leader_pop_dir) in zip(leader_names, leader_pop_dirs)
#             }
    # generations = ["generation"]
    # save_data = []
    

    # else:
    #     trial_path = join("results", computername, "trials", trialname)
    #     # TODO: Figure out how to go through all of the results files and go through directory
    #     [
    #         "evaluation_team": {
    #             "fitnesses": ,  # array of fitnesses 
    #             "joint_trajectory": ,  # joint trajectory as npy array
    #         }
    #         "population": {
    #             "leader_0": ,
    #             "leader_1": ,
    #         }
    #         "training_teams": {
    #             "fitnesses": ,
    #             "joint_trajectories" ,
    #         }
    #     ]

def loadPopulation(trialname: str, computername: Optional[str]) -> List[NN]:
    f = loadTrial(trialname, computername)
    return f["final_population"]


def getLatestTrialNum(computername: Optional[str]) -> int:
    if computername is None:
        computername = getHostName()
    trials_dir = join("results", computername,"trials")

    # Make sure the directory for this computer's results exists
    computer_dir = join("results", computername)
    # print(computer_dir)
    # print(exists(computer_dir))
    if not exists(computer_dir):
        makedirs(computer_dir)
        makedirs(join(computer_dir, "configs"))
        makedirs(join(computer_dir, "trials"))

    filenames = [f for f in listdir(trials_dir) if isfile(join(trials_dir, f)) and f[-4:] == ".pkl" and f[:6] == "trial_"]
    numbers = [int(f[6:-4]) for f in filenames]
    if len(numbers) == 0:
        return -1
    return str(max(numbers))


def getLatestTrialName(computername: Optional[str]) -> str:
    return "trial_" + getLatestTrialNum(computername)


def getNewTrialName(computername: Optional[str]) -> str:
    return "trial_" + str(int(getLatestTrialNum(computername)) + 1)


def generateTrialName(computername: str, trial_num: Optional[str]):
    if trial_num is None:
        trial_name = getNewTrialName(computername)
        trial_num = trial_name.split("_")[1]
    else:
        trial_name = "trial_" + trial_num
    return trial_name

def generateTrialPath(computername: Optional[str], trial_num:Optional[str]):
    if computername is None:
        computername = getHostName()
    trial_name = generateTrialName(computername, trial_num)
    trial_path = join("results", computername, "trials", trial_name)
    return trial_path

def saveTrial(save_data: Dict, config: Dict, computername: Optional[str], trial_num: Optional[str] = None, save_trial_only: bool = False) -> None:
    if computername is None:
        computername = getHostName()
    
    trial_name = generateTrialName(computername=computername, trial_num=trial_num)

    config_path = join("results", computername, "configs")
    trial_path = join("results", computername, "trials")

    # If save_trial_only is false, then we should save the config as well as the trial
    # If save trial_only is true, then we should skip this step and just save the trial
    print(config_path, trial_num)
    if not save_trial_only:
        with open(join(config_path, "config_"+trial_num+".yaml"), "w") as file:
            yaml.dump(config, file)
    
    with open(join(trial_path, trial_name+".pkl"), "wb") as file:
        pickle.dump(save_data, file)

def saveConfig(config: Dict, computername: Optional[str], trial_num: Optional[str] = None, folder_save=False) -> None:
    # If folder_save is true, then it saves the config to the directory for the particular trial
    # Else, it saves the config to a seperate directory that contains all of the configs
    if computername is None:
        computername = getHostName()
    if trial_num is None:
        raise Exception("trial_num needs to be set for saveConfig")
        # trial_name = getNewTrialName(computername)
        # trial_num = trial_name.split("_")[1]

    if folder_save:
        trial_name = "trial_" + trial_num
        config_path = join("results", computername, "trials", trial_name)
        with open(join(config_path, "config.yaml"), "w") as file:
            yaml.dump(config, file)
    else:
        config_path = join("results", computername, "configs")
        with open(join(config_path, "config_"+trial_num+".yaml"), "w") as file:
            yaml.dump(config, file)

def loadConfig(computername: Optional[str]=".", config_name: str = "default.yaml"):
    if computername is None:
        computername = getHostName()

    if computername == ".":
        path = "configs"
    else:
        path = join("results", computername, "configs")
    
    return myaml.safe_load(join(path, config_name))

def setupInitialPopulation(config: Dict):
    if config["load_population"] is not None:
        if config["load_population"] == "latest":
            config["load_population"] = getLatestTrialName()
        return loadPopulation(config["load_population"])
    else:
        return None
