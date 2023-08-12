from typing import Dict, List, Optional
import pickle
from lib.network_lib import NN
from os import listdir, makedirs
from os.path import isfile, join, exists
import yaml
import myaml
import socket

def getHostName():
    return socket.gethostname()

def loadTrial(trialname: str, computername: Optional[str]) -> Dict:
    # By default load the trial from this computer's save directory
    if computername is None:
        computername = getHostName()
    trial_path = join("results", computername, "trials", trialname+".pkl")
    return pickle.load(open(trial_path, "rb"))


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
    if not save_trial_only:
        with open(join(config_path, "config_"+trial_num+".yaml"), "w") as file:
            yaml.dump(config, file)
    
    with open(join(trial_path, trial_name+".pkl"), "wb") as file:
        pickle.dump(save_data, file)

def saveConfig(config: Dict, computername: Optional[str], trial_num: Optional[str] = None) -> None:
    if computername is None:
        computername = getHostName()
    if trial_num is None:
        raise Exception("trial_num needs to be set for saveConfig")
        # trial_name = getNewTrialName(computername)
        # trial_num = trial_name.split("_")[1]
    else:
        trial_name = "trial_" + trial_num

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
