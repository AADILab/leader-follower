"""
@title

@description

"""
import argparse
from pathlib import Path
import time

from leader_follower import project_properties
from leader_follower.environment.boids_env import BoidsEnv
from leader_follower.bak.file_helper import load_config
from leader_follower.learn.cceaV2 import neuro_evolve


def run_experiment(experiment_config):
    env = BoidsEnv(**experiment_config['CCEA']['config']['BoidsEnv'])
    n_hidden = experiment_config['CCEA']['nn_hidden']

    subpop_size = 50
    sim_subpop_size = 15
    n_gens = 100

    start_time = time.time()
    best_solution, max_fits, avg_fits = neuro_evolve(env, n_hidden, subpop_size, n_gens, sim_subpop_size)
    end_time = time.time()

    # rewards = rollout(gw, best_solution)
    # gw.display()
    # print(f'{rewards=}')
    # plot_fitnesses(avg_fitnesses=avg_fits, max_fitnesses=max_fits)
    # gw.plot_agent_trajectories()
    return

def main(main_args):
    subpop_size = 50
    n_gens = 100
    stat_runs = 5
    # config_name = 'default.yaml'

    config_fns = [each_fn for each_fn in Path(project_properties.config_dir).rglob('*.yaml')]
    for each_fn in config_fns:
        print(f'{"=" * 80}')
        print(f'{each_fn}')

        exp_config = load_config(each_fn)
        exp_config['CCEA']['sub_population_size'] = subpop_size
        exp_config['num_generations'] = n_gens

        for idx in range(0, stat_runs):
            run_experiment(exp_config)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
