"""
@title

@description

"""
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from leader_follower import project_properties

def parse_stat_run(stat_run_dir):
    gen_dirs = list(stat_run_dir.glob('gen_*'))
    gen_dirs = sorted(gen_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    generations = []
    for each_dir in gen_dirs:

        fitness_data = []
        fitness_fname = Path(each_dir, 'fitnesses.csv')
        with open(fitness_fname, 'r') as x:
            reader = csv.reader(x, delimiter=',')
            next(reader)
            for row in reader:
                fitness_data.append(row)
        gen_data = {
            each_row[0]: each_row[1:]
            for each_row in fitness_data
        }
        generations.append(gen_data)

    condensed_gens = {name: [] for name in generations[0].keys()}
    for each_gen in generations:
        for each_name, vals in each_gen.items():
            condensed_gens[each_name].append(vals)

    np_gens = {
        name: np.array(data[:880], dtype=float)
        for name, data in condensed_gens.items()
    }
    return np_gens

def parse_experiment_fitnesses(experiment_dir: Path):
    stat_dirs = list(experiment_dir.glob('stat_run_*'))
    stat_dirs = sorted(stat_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    stat_runs = []
    for each_dir in stat_dirs:
        fitness_data = parse_stat_run(each_dir)
        stat_runs.append(fitness_data)
    return stat_runs

def plot_fitnesses(fitness_data, save_dir, config, reward):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    fitness_data = fitness_data[0]
    for reward_label, fitnesses in fitness_data.items():
        max_vals = fitnesses.max(axis=1)
        avg_vals = fitnesses.mean(axis=1)

        # axes.plot(max_vals, label=f'max {reward_label}')
        axes.plot(avg_vals, label=f'avg {reward_label}')

        axes.set_xlabel(f'generation')
        axes.set_ylabel('fitness')

        axes.xaxis.grid()
        axes.yaxis.grid()

        axes.legend(loc='best')

    fig.suptitle(f'{config}: {reward}')

    fig.set_size_inches(7, 5)
    fig.set_dpi(100)

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    plot_name = f'{config}_{reward}'
    save_name = Path(save_dir, plot_name)
    plt.show()
    plt.close()
    # plt.savefig(f'{save_name}.png')
    return

def main(main_args):
    save_dir = Path(project_properties.output_dir, 'experiments', 'figs')
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(project_properties.cached_dir, 'experiments')
    experiment_dirs = base_dir.glob('*_diff_*')

    # experiment_dirs = list(base_dir.glob('*_neuro_evolve_diff_alpha'))
    # experiment_dirs = experiment_dirs[:1]

    for each_dir in experiment_dirs:
        print(f'Processing experiment: {each_dir}')
        exp_name = each_dir.stem
        exp_name = exp_name.split('_')
        exp_name = exp_name[5:]

        reward = exp_name[0]
        config = '_'.join(exp_name[1:])

        fitness_data = parse_experiment_fitnesses(each_dir)
        plot_fitnesses(fitness_data, save_dir=save_dir, config=config, reward=reward)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
