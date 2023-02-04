"""
@title

@description

"""
import argparse
import csv
from pathlib import Path
import os

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
        with open(fitness_fname, 'r') as fitness_file:
            reader = csv.reader(fitness_file, delimiter=',')
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
        name: data
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


def plot_fitnesses(fitness_data, config, save_dir, tag):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    # todo  compute max and std_dev for each stat run rather than just plotting first stat run
    for reward_label, stat_runs in fitness_data.items():
        agent_fitnesses = {}
        for each_run in stat_runs:
            for name, each_fitnesses in each_run.items():
                if name not in agent_fitnesses:
                    agent_fitnesses[name] = []
                agent_fitnesses[name].append(each_fitnesses)

        g_fitnesses = agent_fitnesses['G']
        g_fitnesses = np.asarray(g_fitnesses, dtype=float)

        max_vals = np.max(g_fitnesses, axis=2)
        avg_vals = np.mean(g_fitnesses, axis=2)

        means = np.mean(max_vals, axis=0)
        stds = np.std(max_vals, axis=0)

        gen_idxs = np.arange(0, len(means))
        axes.plot(gen_idxs, means, label=f'max {reward_label}')
        axes.fill_between(gen_idxs, means + stds, means - stds, alpha=0.2)

    axes.set_xlabel(f'generation')
    axes.set_ylabel('fitness')

    axes.xaxis.grid()
    axes.yaxis.grid()

    axes.legend(loc='best')

    fig.suptitle(f'{config}')

    fig.set_size_inches(7, 5)
    fig.set_dpi(100)

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    plot_name = f'{config}'
    save_name = Path(save_dir, f'{plot_name}_{tag}')
    plt.savefig(f'{save_name}.png')
    # plt.show()
    plt.close()
    return


def replay_episode(episode_dir: Path):
    # load saved policies of each agent
    # load environment
    # rollout environment
    stat_dirs = list(episode_dir.glob('stat_run_*'))
    stat_dirs = sorted(stat_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    stat_dirs = [episode_dir]
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
            policy_fnames = sorted(policy_fnames, key=lambda x: int(x.stem.split('_')[-1])) if len(
                policy_fnames) == len(gen_dirs) else policy_fnames

            best_policy_fn = policy_fnames[arg_best_policy] if len(policy_fnames) == len(gen_dirs) else policy_fnames[0]
            model = load_pytorch_model(best_policy_fn)
            agent_policies[agent_name] = {'network': model, 'fitness': best_fitness}

        env_path = Path(each_dir, f'leader_follower_env_initial.pkl')
        env = LeaderFollowerEnv.load_environment(env_path)
        env.render_mode = 'human'
        episode_rewards = rollout(env, agent_policies, LeaderFollowerEnv.calc_diff_rewards,
                                  render={'window_size': 500, 'render_bound': 50})
        g_reward = env.calc_global()
        print(f'stat_run: {idx} | {g_reward=} | {episode_rewards=}')
    return


def main(main_args):
    base_save_dir = Path(project_properties.output_dir, 'experiments', 'figs')
    if not base_save_dir.exists():
        base_save_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(project_properties.cached_dir, 'experiments')
    experiment_dirs = list(base_dir.glob('experiment_*'))

    for each_dir in experiment_dirs:
        print(f'Processing experiment: {each_dir.stem}')
        config_dirs = list(each_dir.glob(f'whiteboardV[1-2]'))
        for config_path in config_dirs:
            config_name = config_path.stem
            print(f'\t{config_path.stem}')
            reward_dirs = list(config_path.glob(f'*'))

            fitnesses = {}
            for reward_path in reward_dirs:
                reward_name = reward_path.stem
                print(f'\t\t{reward_name}')

                fitness_data = parse_experiment_fitnesses(reward_path)
                fitnesses[reward_name] = fitness_data
                replay_episode(reward_path)

            plot_fitnesses(fitnesses, config=config_name, save_dir=base_save_dir, tag=f'{each_dir.stem}_{config_name}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
