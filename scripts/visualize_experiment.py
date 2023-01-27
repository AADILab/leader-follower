"""
@title

@description

"""
import argparse
import csv
from pathlib import Path

from leader_follower import project_properties


def parse_experiment_fitnesses(experiment_dir: Path):
    gen_dirs = list(experiment_dir.glob('gen_*'))
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

    return condensed_gens
def plot_fitnesses(fitness_data):
    return

def main(main_args):
    # todo rewrite to read from experiments in cached directory
    base_dir = Path(project_properties.cached_dir, 'experiments')
    experiment_dirs = base_dir.glob('*_neuro_evolve')
    for each_dir in experiment_dirs:
        print(f'Processing experiment: {each_dir}')
        fitness_data = parse_experiment_fitnesses(each_dir)
        plot_fitnesses(fitness_data)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
