import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional
import pickle
from os import listdir
from os.path import isfile
import yaml
import myaml

from leader_follower.bak.network_lib import NN


def euclidean(positions_a: NDArray[np.float64], positions_b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the distance between positions A and B"""
    return np.linalg.norm(positions_a - positions_b, axis=1)


def calc_delta_heading(current_heading: float, desired_heading: float) -> float:
    """ Calculate delta headings such that delta is the shortest path from
    current heading to the desired heading.
    """
    if desired_heading == current_heading:
        delta_heading = 0
    else:
        # Case 1: Desired heading greater than current heading
        if desired_heading > current_heading:
            desired_heading_prime = desired_heading - 2 * np.pi

        # Case 2: Desired heading less than current heading
        else:
            desired_heading_prime = desired_heading + 2 * np.pi

        delta0 = desired_heading - current_heading
        delta1 = desired_heading_prime - current_heading
        which_delta = np.argmin([np.abs(delta0), np.abs(delta1)])
        delta_heading = np.array([delta0, delta1])[which_delta]
    return delta_heading



def bound_angle(heading, bound=np.pi):
    bounded_heading = heading
    # Bound heading from [0,2pi]
    if bounded_heading > 2 * bound or bounded_heading < 0:
        bounded_heading %= 2 * bound

    # Bound heading from [-pi,+pi]
    if bounded_heading > bound:
        bounded_heading -= 2 * bound
    return bounded_heading


def calc_centroid(positions):
    if positions.size == 0:
        return None
    else:
        return np.average(positions, axis=0)


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def load_trial(trialname: str) -> Dict:
    return pickle.load(open("trials/" + trialname + ".pkl", "rb"))


def load_population(trialname: str) -> List[NN]:
    f = load_trial(trialname)
    return f["final_population"]


def get_latest_trial_num() -> int:
    filenames = [f for f in listdir("trials") if isfile("trials/" + f) and f[-4:] == ".pkl" and f[:6] == "trial_"]
    numbers = [int(f[6:-4]) for f in filenames]
    if len(numbers) == 0:
        return -1
    return str(max(numbers))


def get_latest_trial_name() -> str:
    return "trial_" + get_latest_trial_num()


def get_new_trial_name() -> str:
    return "trial_" + str(int(get_latest_trial_num()) + 1)


def save_trial(save_data: Dict, config: Dict, trial_num: Optional[str] = None) -> None:
    if trial_num is None:
        trial_name = get_new_trial_name()
        trial_num = trial_name.split("_")[1]
    else:
        trial_name = "trial_" + trial_num
    with open("configs/config_" + trial_num + ".yaml", "w") as file:
        yaml.dump(config, file)
    pickle.dump(save_data, open("trials/" + trial_name + ".pkl", "wb"))


def load_config(config_name: str = "default.yaml"):
    return myaml.safe_load("configs/" + config_name)


def setup_initial_population(config: Dict):
    if config["load_population"] is not None:
        if config["load_population"] == "latest":
            config["load_population"] = get_latest_trial_name()
        return load_population(config["load_population"])
    else:
        return None
