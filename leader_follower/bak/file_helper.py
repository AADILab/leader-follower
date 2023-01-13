import pickle
from pathlib import Path
from typing import Dict, List, Optional

import myaml

from leader_follower.bak.network_lib import NN


def load_trial(base_dir, trial_name: str) -> Dict:
    trial_path = Path(base_dir, 'trials', trial_name)
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
        # trial_num = trial_name.split('_')[1]
    else:
        trial_name = f'trial_{trial_num}'

    # configs should not change between trials
    # config_name = f'config_{trial_num}.yaml'
    # config_path = Path(base_dir, 'configs', config_name)
    # if not config_path.parent.exists() or not config_path.parent.is_dir():
    #     config_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(config_path, 'w') as file:
    #     yaml.dump(config, file)

    # todo save as json for readability
    trial_name = f'{trial_name}.pkl'
    trial_path = Path(base_dir, 'trials', trial_name)
    if not trial_path.parent.exists() or not trial_path.parent.is_dir():
        trial_path.parent.mkdir(parents=True, exist_ok=True)

    with open(trial_path, 'wb') as file:
        pickle.dump(save_data, file)

    return trial_path


def load_config(config_name):
    config_path = str(Path(config_name))
    return myaml.safe_load(config_path)


def setup_initial_population(base_dir, meta_params):
    if meta_params['load_population'] is None:
        return None

    if meta_params["load_population"] == "latest":
        meta_params["load_population"] = latest_trial_name(base_dir)
    return load_population(base_dir, meta_params["load_population"])
