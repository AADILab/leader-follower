from typing import Dict, List, Optional
import pickle
from lib.network_lib import NN
from os import listdir
from os.path import isfile
import yaml
import myaml


def loadTrial(trialname: str) -> Dict:
    return pickle.load(open("trials/" + trialname + ".pkl", "rb"))


def loadPopulation(trialname: str) -> List[NN]:
    f = loadTrial(trialname)
    return f["final_population"]


def getLatestTrialNum() -> int:
    filenames = [f for f in listdir("trials") if isfile("trials/" + f) and f[-4:] == ".pkl" and f[:6] == "trial_"]
    numbers = [int(f[6:-4]) for f in filenames]
    if len(numbers) == 0:
        return -1
    return str(max(numbers))


def getLatestTrialName() -> str:
    return "trial_" + getLatestTrialNum()


def getNewTrialName() -> str:
    return "trial_" + str(int(getLatestTrialNum()) + 1)


def saveTrial(save_data: Dict, config: Dict, trial_num: Optional[str] = None) -> None:
    if trial_num is None:
        trial_name = getNewTrialName()
        trial_num = trial_name.split("_")[1]
    else:
        trial_name = "trial_" + trial_num
    with open("configs/config_" + trial_num + ".yaml", "w") as file:
        yaml.dump(config, file)
    pickle.dump(save_data, open("trials/" + trial_name + ".pkl", "wb"))


def loadConfig(config_name: str = "default.yaml"):
    return myaml.safe_load("configs/" + config_name)


def setupInitialPopulation(config: Dict):
    if config["load_population"] is not None:
        if config["load_population"] == "latest":
            config["load_population"] = getLatestTrialName()
        return loadPopulation(config["load_population"])
    else:
        return None
