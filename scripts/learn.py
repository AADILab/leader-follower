from pathlib import Path

from leader_follower.file_helper import load_config
from leader_follower.learn_helpers import run_experiment
from leader_follower.project_properties import data_dir


def main():
    experiments = [
        Path(data_dir, 'alpha', 'all_leaders', 'difference_leader'),
        # Path(data_dir, 'alpha', 'all_leaders', 'difference_leader_follower'),
        # Path(data_dir, 'alpha', 'all_leaders', 'global'),
        #
        # Path(data_dir, 'alpha', 'leaders_followers', 'difference_leader'),
        # Path(data_dir, 'alpha', 'leaders_followers', 'difference_leader_follower'),
        # Path(data_dir, 'alpha', 'leaders_followers', 'global'),
        #
        # Path(data_dir, 'charlie', 'all_leaders', 'difference_leader'),
        # Path(data_dir, 'charlie', 'all_leaders', 'difference_leader_follower'),
        # Path(data_dir, 'charlie', 'all_leaders', 'global'),
        #
        # Path(data_dir, 'charlie', 'leaders_followers', 'difference_leader'),
        # Path(data_dir, 'charlie', 'leaders_followers', 'difference_leader_follower'),
        # Path(data_dir, 'charlie', 'leaders_followers', 'global'),
        #
        # Path(data_dir, 'echo', 'all_leaders', 'difference_leader'),
        # Path(data_dir, 'echo', 'all_leaders', 'difference_leader_follower'),
        # Path(data_dir, 'echo', 'all_leaders', 'global'),
        #
        # Path(data_dir, 'echo', 'leaders_followers', 'difference_leader'),
        # Path(data_dir, 'echo', 'leaders_followers', 'difference_leader_follower'),
        # Path(data_dir, 'echo', 'leaders_followers', 'global'),
    ]
    config_name = 'default.yaml'
    subpop_size = 50
    n_gens = 100
    stat_runs = 5
    for each_experiment in experiments:
        # try:
        print(f'{"=" * 80}')
        print(f'{each_experiment}')
        config = load_config(each_experiment, config_name=config_name)
        config['CCEA']['sub_population_size'] = subpop_size
        config['num_generations'] = n_gens
        # todo get stat_runs from config
        # Run each experiment n times
        for idx in range(stat_runs):
            print(f'Running experiment {idx}')
            run_experiment(each_experiment, config)
        print(f'{"=" * 80}')
        # except Exception as e:
        #     print(f'Experiment {each_experiment}\n'
        #           f'{e}')
        #     print(f'{"=" * 80}')
    return


if __name__ == '__main__':
    main()
