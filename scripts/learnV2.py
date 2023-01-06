"""
@title

@description

"""
import argparse
from pathlib import Path
import time

from leader_follower import project_properties
from leader_follower.file_helper import load_config


def run_experiment(experiment_config):
    # Start clock
    start_time = time.time()
    end_time = time.time()

    return

def main(main_args):
    subpop_size = 50
    n_gens = 100
    stat_runs = 5
    config_name = 'default.yaml'

    config_fns = [each_fn for each_fn in Path(project_properties.config_dir).rglob('*.yaml')]
    for each_fn in config_fns:
        print(f'{"=" * 80}')
        print(f'{each_fn}')

        config = load_config(each_fn.parent, config_name=config_name)
        config['CCEA']['sub_population_size'] = subpop_size
        config['num_generations'] = n_gens

        for idx in range(0, stat_runs):
            run_experiment(each_fn)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
