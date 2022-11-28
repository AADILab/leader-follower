import pickle
from pathlib import Path
from typing import Dict, List, Optional

import myaml
import yaml

from leader_follower.network_lib import NN
from leader_follower.project_properties import data_dir


def load_trial(base_dir, trial_name: str) -> Dict:
    trial_path = Path(base_dir, 'trials', f'{trial_name}.pkl')
    with open(trial_path, 'rb') as trial_file:
        trial_data = pickle.load(trial_file)
    return trial_data


def load_population(base_dir, trial_name: str) -> List[NN]:
    trial_data = load_trial(base_dir, trial_name)
    return trial_data['final_population']


def latest_trial_num(base_dir) -> int:
    trials_dir = Path(base_dir, 'trials')
    trial_nums = [
        int(each_file.stem.split('_')[-1])
        for each_file in trials_dir.glob('*.pkl')
        if each_file.is_file() and each_file.suffix == '.pkl' and each_file.stem.startswith('trial_')
    ]
    if len(trial_nums) == 0:
        return -1
    return max(trial_nums)


def latest_trial_name(base_dir):
    return f'trial_{latest_trial_num(base_dir)}'


def new_trial_name(base_dir) -> str:
    return f'trial_{int(latest_trial_num(base_dir)) + 1}'


def save_trial(base_dir, save_data: Dict, config: Dict, trial_num: Optional[str] = None):
    if trial_num is None:
        trial_name = new_trial_name(base_dir)
        trial_num = trial_name.split('_')[1]
    else:
        trial_name = f'trial_{trial_num}'

    trial_name = f'{trial_name}.pkl'
    config_name = f'config_{trial_num}.yaml'
    config_path = Path(base_dir, 'configs', config_name)
    trial_path = Path(base_dir, 'trials', trial_name)

    # with open("configs/config_" + trial_num + ".yaml", "w") as file:
    # todo configs should not change trial to trial
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    with open(trial_path, 'wb') as file:
        pickle.dump(save_data, file)
        # pickle.dump(save_data, open("trials/" + trial_name + ".pkl", "wb"))
    return config_path, trial_path


def load_config(base_dir: data_dir, config_name: str = 'default.yaml'):
    return myaml.safe_load(str(Path(base_dir, config_name)))
    # return myaml.safe_load("configs/" + config_name)


def setup_initial_population(base_dir, config: Dict):
    if config['load_population'] is None:
        return None

    if config["load_population"] == "latest":
        config["load_population"] = latest_trial_name(base_dir)
    return load_population(base_dir, config["load_population"])
