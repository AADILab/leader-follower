from pathlib import Path

from leader_follower.file_helper import load_config
from leader_follower.learn_helpers import run_experiment
from leader_follower.project_properties import data_dir


def main():
    n_exp = 2
    # Run each experiment n times
    for _ in range(n_exp):
        base_dir = Path(data_dir, 'alpha', 'leaders_followers')
        config_name = 'default.yaml'
        config = load_config(base_dir, config_name=config_name)
        config["CCEA"]["use_difference_evaluations"] = True
        run_experiment(base_dir, config)
    return


if __name__ == '__main__':
    main()
