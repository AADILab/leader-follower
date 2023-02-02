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
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.cceaV2 import rollout
from leader_follower.learn.neural_network import load_pytorch_model


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
        name: np.array(data, dtype=float)
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

    # todo  compute max and std_dev for each stat run rather than just plotting first stat run
    fitness_data = fitness_data[0]
    for reward_label, fitnesses in fitness_data.items():
        max_vals = fitnesses.max(axis=1)
        avg_vals = fitnesses.mean(axis=1)

        axes.plot(max_vals, label=f'max {reward_label}')
        # axes.plot(avg_vals, label=f'avg {reward_label}')

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
    # plt.savefig(f'{save_name}.png')
    plt.close()
    return

def replay_episode(episode_dir: Path):
    # load saved policies of each agent
    # load environment
    # rollout environment
    stat_dirs = list(episode_dir.glob('stat_run_*'))
    stat_dirs = sorted(stat_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    for idx, each_dir in enumerate(stat_dirs):
        fitness_data = parse_stat_run(each_dir)
        gen_dirs = list(each_dir.glob('gen_*'))
        gen_dirs = sorted(gen_dirs, key=lambda x: int(x.stem.split('_')[-1]))
        last_gen_idx = len(gen_dirs) - 1
        last_gen_dir = gen_dirs[last_gen_idx]

        agent_policies = {}
        policy_dirs = last_gen_dir.glob(f'*_networks')
        for agent_policy_dir in policy_dirs:
            agent_name = agent_policy_dir.stem.split('_')
            agent_name = '_'.join(agent_name[:2])

            # find the best policies for each agent based on fitnesses.csv
            agent_fitnesses = fitness_data[agent_name][last_gen_idx]
            arg_best_policy = np.argmax(agent_fitnesses)
            best_fitness = agent_fitnesses[arg_best_policy]

            policy_fnames = list(agent_policy_dir.glob(f'*_model*.pt'))
            policy_fnames = sorted(policy_fnames, key=lambda x: int(x.stem.split('_')[-1])) if len(policy_fnames) == len(gen_dirs) else policy_fnames

            best_policy_fn = policy_fnames[arg_best_policy] if len(policy_fnames) == len(gen_dirs) else policy_fnames[0]
            model = load_pytorch_model(best_policy_fn)
            agent_policies[agent_name]= {'network': model, 'fitness': best_fitness}

        env_path = Path(each_dir, f'leader_follower_env_initial.pkl')
        env = LeaderFollowerEnv.load_environment(env_path)
        env.render_mode = 'human'
        episode_rewards = rollout(env, agent_policies, LeaderFollowerEnv.calc_diff_rewards, render=True)
        g_reward = env.calc_global()
        print(f'stat run: {idx}: {g_reward=}: {episode_rewards=}')
    return

def main(main_args):
    save_dir = Path(project_properties.output_dir, 'experiments', 'figs')
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(project_properties.cached_dir, 'experiments')

    experiment_dirs = base_dir.glob('2023_02_01_20_08_difflf_atrium')
    # experiment_dirs = base_dir.glob('*_diff*_atrium')
    # experiment_dirs = base_dir.glob('*_diff_whiteboardV*')
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
        # plot_fitnesses(fitness_data, save_dir=save_dir, config=config, reward=reward)
        replay_episode(each_dir)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
