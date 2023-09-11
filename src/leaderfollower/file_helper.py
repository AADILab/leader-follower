from typing import Dict, List, Optional
import pickle
from leaderfollower.network_lib import NN
from leaderfollower.data_helpers import getTrialNames
from os import listdir, makedirs
from os.path import isfile, join, exists
import yaml
import myaml
import socket
import numpy as np
import tqdm

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

def loadTrialData(trialname: str, computername: Optional[str], load_populations=True, load_evaluation_teams=True, load_training_teams=True):
    """This loads in the data for a trial assuming there is a folder for the trial with an npz file for each generation"""
    if computername is None:
        computername = getHostName()

    trial_path = join("results", computername, "trials", trialname)

    # Set up list of generation npz files so we can traverse them 
    files_list = listdir(path=trial_path)
    generations_list = [file for file in files_list if file[:11]=="generation_"]
    unsorted_generation_npz_dirs = [join(trial_path, generation_file) for generation_file in generations_list]
    gen_nums = [int(gen[11:-4]) for gen in generations_list]
    generation_npz_dirs = [dir for _, dir in sorted(zip(gen_nums, unsorted_generation_npz_dirs))]

    # Go through each generation and store relevant information
    trial_data = []
    for generation_npz in tqdm.tqdm(generation_npz_dirs):
        generation_dict = {}

        loaded_npz = np.load(generation_npz)

        if load_evaluation_teams:
            # Get the evaluation team information first
            generation_dict["evaluation_team"] = {
                "team_fitness" : loaded_npz["evaluation_team|team_fitness"][0],
                "agent_fitnesses" : loaded_npz["evaluation_team|agent_fitnesses"],
                "policy_ids" : loaded_npz["evaluation_team|policy_ids"],
                "joint_trajectory" : loaded_npz["evaluation_team|joint_trajectory"]
            }

        if load_training_teams:
            # Get all of the training team information

            # Figure out the files in the npz
            npy_files = loaded_npz.files
            # Filter out files that are only related to the training teams
            training_teams_files = [file for file in npy_files if file[:14] == "training_teams"]
            
            # Get the team names
            team_names = set([file.split("|")[1] for file in training_teams_files])
            # Now go through each team and get the appropriate information
            generation_dict["training_teams"] = {}
            for team_name in team_names:
                generation_dict["training_teams"][team_name] = {
                    "team_fitness" : loaded_npz["training_teams|"+team_name+"|team_fitness"][0],
                    "agent_fitnesses" : loaded_npz["training_teams|"+team_name+"|agent_fitnesses"],
                    "policy_ids" : loaded_npz["training_teams|"+team_name+"|policy_ids"],
                    "joint_trajectory" : loaded_npz["training_teams|"+team_name+"|joint_trajectory"]
                }

        if load_populations:
            # Go through all the agent populations
            
            # Figure out how many leaders there are
            population_files = [file for file in npy_files if file[:10] == "population"]
            leader_names = set([file.split("|")[1] for file in population_files])

            # Go through the leaders and add their policies
            generation_dict["population"] = {}

            for leader_name in leader_names:
                generation_dict["population"][leader_name] = {}

                # Figure out how many policies this leader has in its population
                policy_files = [file for file in population_files if file.split("|")[1] == leader_name]
                policy_names = set([file.split("|")[2] for file in policy_files])

                # Figure out how many layers each policy had
                # Assume the layers for each policy are the same number and size
                # These are the layer files for just the 0th policy for leader 0
                layer_files = [file for file in policy_files if file.split("|")[3][:5] == "layer"]
                layer_names = set([file.split("|")[3] for file in layer_files])

                # Save those policies
                for policy_name in policy_names:
                    generation_dict["population"][leader_name][policy_name] = {}
                    for layer_name in layer_names:
                        generation_dict["population"][leader_name][policy_name][layer_name] = loaded_npz["population|"+leader_name+"|"+policy_name+"|"+layer_name]
        
        trial_data.append(generation_dict)
    
    return trial_data

def loadMultiTrialsData(trialnames: List[str], computername: str, load_populations = True, load_evaluation_teams=True, load_training_teams=True):
    """A batch is a set of trials that were all run with exactly the same parameters. They are a subset of trials run in an experiment (computer) folder. A batch can include different variants of reward shaping though"""
    batch_data = []
    for trialname in trialnames:
        batch_data.append(loadTrialData(trialname=trialname,computername=computername, load_populations=load_populations, load_evaluation_teams=load_evaluation_teams, load_training_teams=load_training_teams))
    return batch_data

def loadExperimentData(computername: str, load_populations = True, load_evaluation_teams=True, load_training_teams=True):
    """This just loads in all of the trials from a particular experiment (computername)"""
    trialnames = listdir(join("results", computername))
    return loadBatchData(trialnames=trialnames, load_populations=load_populations, load_evaluation_teams=load_evaluation_teams, load_training_teams=load_training_teams)

def loadTrialDataMultiFile(trialname: str, computername: Optional[str]):
    """This is legacy code for a brief setup I had where each generation had several npz files saving seperate pieces of data"""
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
        if not exists(config_path): makedirs(name=config_path)
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

def loadConfigDir(config_dir: str):
    """Loads a config from a specified path"""

    return myaml.safe_load(config_dir)

def loadConfigData(trialname: str, computername: Optional[str]) -> Dict:
    if computername is None:
        computername = getHostName()
    trial_path = join("results", computername, "trials", trialname)

    # Get the config file
    config_dir = join(trial_path, "config.yaml")

    # Load it in as a yaml file
    return myaml.safe_load(config_dir)

def setupInitialPopulation(config: Dict):
    if config["load_population"] is not None:
        if config["load_population"] == "latest":
            config["load_population"] = getLatestTrialName()
        return loadPopulation(config["load_population"])
    else:
        return None

def loadBatch(computername: str, start_trial_num: int, num_stat_runs: int, tested_G: bool, tested_D: bool, tested_Dfollow: bool):
    # Generate trial names
    trial_num = start_trial_num
    if tested_Dfollow: 
        trials_Dfollow, trial_num = getTrialNames(trial_num, num_stat_runs)
        print("Dfollow trials: ", trials_Dfollow)

    if tested_D: 
        trials_D, trial_num = getTrialNames(trial_num, num_stat_runs)
        print("D trials: ", trials_D)

    if tested_G: 
        trials_G, trial_num = getTrialNames(trial_num, num_stat_runs)
        print("G trials: ", trials_G)

    # Load in those trials
    if tested_Dfollow: 
        trial_datas_Dfollow = loadMultiTrialsData(trialnames=trials_Dfollow, computername=computername, load_populations=False, load_evaluation_teams=True, load_training_teams=True)
        num_generations = len(trial_datas_Dfollow[0])
    else:
        trial_datas_Dfollow = None
    if tested_D: 
        trial_datas_D = loadMultiTrialsData(trialnames=trials_D, computername=computername, load_populations=False, load_evaluation_teams=True, load_training_teams=True)
        num_generations = len(trial_datas_D[0])
    else:
        trial_datas_D = None
    if tested_G: 
        trial_datas_G = loadMultiTrialsData(trialnames=trials_G, computername=computername, load_populations=False, load_evaluation_teams=True, load_training_teams=True)
        num_generations = len(trial_datas_G[0])
    else:
        trial_datas_G = None
    
    return num_generations, trial_datas_Dfollow, trial_datas_D, trial_datas_G