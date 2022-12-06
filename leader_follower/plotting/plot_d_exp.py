# Configs 247 - 291. Sweep of different numbers of learners to followers
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from leader_follower import project_properties
from leader_follower.file_helper import load_config, load_trial
from leader_follower.project_properties import output_dir


def process_experiment(exp_dir, start_trial, stat_runs):
    reward_types = [
        'difference_leader',
        'difference_leader_follower',
        'global'
    ]
    markers = ['o', '^', 'p']

    configs = [Path('configs', f'config_{i + start_trial}.yaml') for i in range(stat_runs)]
    trials = [f'trial_{i + start_trial}.pkl' for i in range(stat_runs)]

    data_dict = {}
    for each_reward in reward_types:
        reward_dir = Path(exp_dir, each_reward)

        # Get the average best performance of each one
        loaded_trials = [load_trial(reward_dir, trial) for trial in trials]
        loaded_configs = [load_config(reward_dir, config) for config in configs]

        # This will hold average team fitnesses from CCEA population.
        # For each trial, we have the average team fitness for each generation
        all_team_fitnesses = []
        data_dict[each_reward] = {}
        for save_data, loaded_config, config in zip(loaded_trials, loaded_configs, configs):
            unfiltered_scores_list = save_data['unfiltered_scores_list']
            all_team_fitnesses.append(unfiltered_scores_list)
        all_team_fitness_arr = np.array(all_team_fitnesses)
        data_dict[each_reward]['all_team_fitness_arr'] = all_team_fitness_arr
        data_dict[each_reward]['avg_team_fitness_arr'] = np.average(all_team_fitness_arr, axis=0)
        data_dict[each_reward]['max_team_fitness_arr'] = np.max(all_team_fitness_arr, axis=0)
        data_dict[each_reward]['std_dev_team_fitness_arr'] = np.std(all_team_fitness_arr, axis=0)
        data_dict[each_reward]['upper_std_dev_team_fitness_arr'] = data_dict[each_reward]['avg_team_fitness_arr'] + \
            data_dict[each_reward]['std_dev_team_fitness_arr']
        data_dict[each_reward]['lower_std_dev_team_fitness_arr'] = data_dict[each_reward]['avg_team_fitness_arr'] - \
            data_dict[each_reward]['std_dev_team_fitness_arr']
        data_dict[each_reward]['upper_range'] = np.max(data_dict[each_reward]['avg_team_fitness_arr'], axis=0)
        data_dict[each_reward]['lower_range'] = np.min(data_dict[each_reward]['avg_team_fitness_arr'], axis=0)

    # Plot data
    # # todo use project_properties
    reward_dir = Path(exp_dir, reward_types[0])
    query_config = load_config(reward_dir, configs[0])
    num_generations_arr = np.arange(query_config['num_generations'] + 1)

    plt.figure(0)
    plt.ylim([-0.1, 1.0])
    for each_comp, m in zip(reward_types, markers):
        plt.plot(num_generations_arr, data_dict[each_comp]['avg_team_fitness_arr'], marker=m)
        # plt.plot(num_generations_arr, data_dict[each_comp]['max_team_fitness_arr'], marker=m)
    plt.legend(reward_types)
    for each_comp in reward_types:
        plt.fill_between(num_generations_arr, data_dict[each_comp]['upper_std_dev_team_fitness_arr'],
                         data_dict[each_comp]['lower_std_dev_team_fitness_arr'], alpha=0.2)

    experiment = exp_dir.stem
    configuration_type = exp_dir.parent.stem
    plt.title(f'Different Learning Methods: {configuration_type} with {experiment.replace("_", " ")}')
    plt.xlabel('Number of Generations')
    plt.ylabel('Average Team Fitness Score')
    plt.tight_layout()
    plt.savefig(Path(output_dir, f'{configuration_type}_{experiment}'))
    plt.close()
    return


def main():
    # Data Wrangling
    ##
    start_trial = 5
    stat_runs = 5

    exp_configs = [
        'alpha',
        'charlie',
        'echo',
    ]
    base_dirs = [
        Path(project_properties.data_dir, each_exp)
        for each_exp in exp_configs
    ]

    exp_types = [
        'all_leaders',
        'leaders_followers'
    ]

    for exp_dir in base_dirs:
        for each_type in exp_types:
            exp_path = Path(exp_dir, each_type)
            process_experiment(exp_path, start_trial, stat_runs)
    return


if __name__ == '__main__':
    main()
