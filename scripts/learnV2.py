"""
@title

@description

"""
import argparse
import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import trange

from leader_follower import project_properties
from leader_follower.agent import Leader, Follower, Poi, AgentType
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.cceaV2 import neuro_evolve, rollout
from leader_follower.learn.neural_network import NeuralNetwork
from leader_follower.utils import load_config

reward_map = {
    'global': LeaderFollowerEnv.calc_global,

    'diff': partial(LeaderFollowerEnv.calc_diff_rewards, **{'remove_followers': False}),
    'difflf': partial(LeaderFollowerEnv.calc_diff_rewards, **{'remove_followers': True}),

    # 'dpp': partial(LeaderFollowerEnv.calc_dpp, **{'remove_followers': False}),
    # 'dpplf': partial(LeaderFollowerEnv.calc_dpp, **{'remove_followers': True})
}

def run_experiment(experiment_config, meta_vars):
    reward_func = reward_map[meta_vars['reward_key']]
    # todo  add noise to location of agents
    leaders = [
        Leader(
            agent_id=idx, location=each_pos, sensor_resolution=meta_vars['sensor_resolution'], value=meta_vars['leader_value'],
            max_velocity=meta_vars['leader_max_velocity'], weight=meta_vars['leader_weight'],
            observation_radius=meta_vars['leader_obs_rad'], policy=None)
        for idx, each_pos in enumerate(experiment_config['leader_positions'])
    ]
    followers = [
        Follower(
            agent_id=idx, location=each_pos, sensor_resolution=meta_vars['sensor_resolution'], value=meta_vars['follower_value'],
            max_velocity=meta_vars['follower_max_velocity'], weight=meta_vars['follower_weight'],
            repulsion_radius=meta_vars['repulsion_rad'], repulsion_strength=meta_vars['repulsion_strength'],
            attraction_radius=meta_vars['attraction_rad'], attraction_strength=meta_vars['attraction_strength'])
        for idx, each_pos in enumerate(experiment_config['follower_positions'])
    ]
    pois = [
        Poi(agent_id=idx, location=each_pos, sensor_resolution=meta_vars['sensor_resolution'], value=meta_vars['poi_value'],
            weight=meta_vars['poi_weight'],
            observation_radius=meta_vars['poi_obs_rad'], coupling=meta_vars['poi_coupling'])
        for idx, each_pos in enumerate(experiment_config['poi_positions'])
    ]
    env = LeaderFollowerEnv(leaders=leaders, followers=followers, pois=pois, max_steps=meta_vars['episode_length'])

    ######################################################
    # todo  allow for policy sharing
    agent_pops = {
        agent_name: [
            {
                'network': NeuralNetwork(
                    n_inputs=env.agent_mapping[agent_name].n_in,
                    n_hidden=meta_vars['n_hidden_layers'],
                    n_outputs=env.agent_mapping[agent_name].n_out,
                ),
                'fitness': None
            }
            for _ in range(meta_vars['population_size'],)
        ]
        for agent_name in env.agents
        if env.agent_mapping[agent_name].type == AgentType.Learner
    }

    # initial fitness evaluation of all networks in population
    print(f'Initializing fitness values for networks')
    for pop_idx in trange(meta_vars['population_size']):
        new_inds = {agent_name: policy_info[pop_idx] for agent_name, policy_info in agent_pops.items()}
        agent_rewards = rollout(env, new_inds, reward_func=reward_func, render=False)
        for agent_name, policy_info in agent_pops.items():
            policy_fitness = agent_rewards[agent_name]
            policy_info[pop_idx]['fitness'] = policy_fitness
    ########################################################
    print(f'Starting experiment: {meta_vars["config_name"]} | {meta_vars["reward_key"]}')
    start_time = time.time()
    best_solution = neuro_evolve(
        env, agent_pops, meta_vars['population_size'], meta_vars['n_gens'], meta_vars['num_simulations'],
        reward_func=reward_func, experiment_dir=meta_vars['experiment_dir']
    )
    end_time = time.time()
    print(f'Time to train: {end_time - start_time}')

    rewards = rollout(env, best_solution, reward_func=reward_func)
    print(f'{rewards=}')
    return

def run_parameter_sweep(base_dir, stat_runs, experiment_config, meta_vars, **parameters):
    n_params = len(parameters)
    if n_params == 0:
        for reward_key in reward_map.keys():
            reward_path = Path(base_dir, f'{reward_key}')
            if not reward_path.exists():
                reward_path.mkdir(parents=True, exist_ok=True)

            meta_vars['reward_key'] = reward_key

            for idx in range(0, stat_runs):
                stat_path = Path(reward_path, f'stat_run_{idx}')
                if not stat_path.exists():
                    stat_path.mkdir(parents=True, exist_ok=True)

                meta_vars['experiment_dir'] = str(stat_path)
                meta_fname = Path(stat_path, f'meta_vars.json')

                with open(meta_fname, 'w') as jfile:
                    json.dump(meta_vars, jfile, indent=2)

                run_experiment(experiment_config=experiment_config, meta_vars=meta_vars)
    else:
        param_keys = list(parameters.keys())
        first_key = param_keys[0]
        param_vals = parameters.pop(first_key)

        for val in np.arange(*param_vals):
            param_dir = Path(base_dir, f'{first_key}_{val}')
            if not param_dir.exists():
                param_dir.mkdir(parents=True, exist_ok=True)

            meta_vars[first_key] = val
            run_parameter_sweep(param_dir, stat_runs, experiment_config, meta_vars, **parameters)
    return

def main(main_args):
    config_names = [
        # 'whiteboardV1',
        # 'whiteboardV1_all_leaders',
        # 'whiteboardV2',
        # 'whiteboardV2_all_leaders',
        'alpha',
        # 'atrium',
        # 'battery',
        # 'charlie',
        # 'echo'
    ]

    config_fns = [
        each_fn
        for each_fn in Path(project_properties.config_dir).rglob('*.yaml')
        if each_fn.stem in config_names
    ]
    stat_runs = 3

    meta_vars = {
        'n_hidden_layers': 2,

        'leader_obs_rad': 100,
        # leader and follower values determine have much "observational power" an agent has
        'leader_value': 1,
        'follower_value': 1,
        # poi value determines how much it is worth to capture this POI
        'poi_value': 1,
        'poi_weight': 0,

        #########################################
        # the below are things that likely have to be fine-tuned for good results on any given configuration
        # 'population_size': 55,
        # 'num_simulations': 25,
        # 'n_gens': 50,
        # 'episode_length': 75,
        # 'sensor_resolution': 8,
        'population_size': 5,
        'num_simulations': 5,
        'n_gens': 5,
        'episode_length': 5,
        'sensor_resolution': 4,

        # leaders have a higher weight to allow for followers to be attracted to leaders more than followers
        'follower_weight': 0.5,
        'leader_weight': 5,

        'repulsion_rad': 0.5,
        'attraction_rad': 3,
        'repulsion_strength': 3,
        'attraction_strength': 0.5,

        'leader_max_velocity': 3,
        'follower_max_velocity': 0.75,

        'poi_obs_rad': 2,
        'poi_coupling': 3,

        'config_name': None,
        'experiment_config': None,
        'reward_key': None,
        'experiment_dir': None,
    }
    now = datetime.now()
    experiment_id = f'experiment_{now.strftime("%Y_%m_%d_%H_%M_%S")}'
    exp_path = Path(project_properties.cached_dir, 'experiments', f'{experiment_id}')
    if not exp_path.exists():
        exp_path.mkdir(parents=True, exist_ok=True)

    for each_fn in config_fns:
        print(f'{"=" * 80}')
        print(f'{each_fn}')

        experiment_config = load_config(each_fn)
        config_name = each_fn.stem
        meta_vars['config_name'] = config_name
        meta_vars['experiment_config'] = experiment_config

        config_experiment_dir = Path(exp_path, f'{config_name}')
        if not config_experiment_dir.exists():
            config_experiment_dir.mkdir(parents=True, exist_ok=True)

        sweep_params = {
            # 'population_size': (25, 55, 15),
            # 'num_simulations': (25, 55, 15),
            # 'n_gens': (100, 1000, 100),
            # 'episode_length': (50, 150, 50),

            # 'sensor_resolution': (4, 8, 4),

            # leaders have a higher weight to allow for followers to be attracted to leaders more than followers
            'follower_weight': (0.5, 2.5, 0.5),
            'leader_weight': (1, 8.5, 2.5),

            'repulsion_rad': (0.5, 3.5, 1.5),
            'attraction_rad': (2.5, 7.5, 2.5),
            'repulsion_strength': (2, 8, 2),
            'attraction_strength': (0.5, 2, 0.5),

            # 'leader_max_velocity': 3,
            # 'follower_max_velocity': 0.75,

            # 'poi_obs_rad': (2, 15, 3),
            # 'poi_coupling': (3, 15, 3),
        }

        run_parameter_sweep(
            base_dir=config_experiment_dir, stat_runs=stat_runs,
            experiment_config=experiment_config, meta_vars=meta_vars,
            **sweep_params
        )
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
