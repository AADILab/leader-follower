from typing import Dict, List, Optional
import pickle
from lib.network_lib import NN
from os import listdir
from os.path import isfile

def loadTrial(trialname: str)->Dict:
    return pickle.load(open("trials/"+trialname+".pkl", "rb"))

def loadPopulation(trialname:str)->List[NN]:
    f = loadTrial(trialname)
    return f["final_population"]

def getLatestTrialNum()->int:
    filenames = [f for f in listdir("trials") if isfile("trials/"+f) and f[-4:] == ".pkl" and f[:6] == "trial_"]
    numbers = [int(f[6:-4]) for f in filenames]
    return str(max(numbers))

def getLatestTrialName()->str:
    return "trial_"+getLatestTrialNum()

def getNewTrialName()->str:
    return "trial_"+str(int(getLatestTrialNum())+1)

def saveTrial(save_data: Dict, trial_num: Optional[str]=None)->None:
    if trial_num is None:
        trial_name = getNewTrialName()
    else:
        trial_name = "trial_"+trial_num
    pickle.dump(save_data, open("trials/"+trial_name+".pkl", "wb"))
