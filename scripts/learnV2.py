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

from leader_follower import project_properties
from leader_follower.agent import Leader, Follower, Poi, AgentType
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.cceaV2 import neuro_evolve, rollout
from leader_follower.learn.neural_network import NeuralNetwork
from leader_follower.learn.rewards import calc_diff_rewards
from leader_follower.utils import load_config

reward_map = {
    # 'global': calc_global,

    'diff': partial(calc_diff_rewards, **{'remove_followers': False}),
    'difflf': partial(calc_diff_rewards, **{'remove_followers': True}),

    # 'dpp': partial(calc_dpp, **{'remove_followers': False}),
    # 'dpplf': partial(calc_dpp, **{'remove_followers': True})
}

def run_experiment(experiment_config, experiment_dir):
    config_parts = experiment_dir.parent.stem.split('_')
    reward_key = config_parts[5]
    config_name = config_parts[6:]
    config_name = '_'.join(config_name)

    meta_vars = {
        'n_hidden_layers': 2,

        'population_size': 35,
        'sim_pop_size': 35,
        'n_gens': 150,
        'episode_length': 50,
        'sensor_resolution': 4,

        'leader_obs_rad': 100,
        'leader_value': 1,
        'leader_max_velocity': 3,

        'follower_value': 1,
        'follower_max_velocity': 1,
        'repulsion_rad': 0.5,
        'repulsion_strength': 5,
        'attraction_rad': 3,
        'attraction_strength': 1,

        'poi_obs_rad': 1,
        'poi_value': 0,
        'poi_coupling': 1,

        'config_name': config_name,
        'reward_key': reward_key,
        'experiment_dir': str(experiment_dir),
        'experiment_config': experiment_config
    }

    meta_fname = Path(experiment_dir, f'meta_vars.json')

    with open(meta_fname, 'w') as jfile:
        json.dump(meta_vars, jfile, indent=2)
    #####################################################################
    reward_func = reward_map[reward_key]
    # todo  add noise to location of agents
    leaders = [
        Leader(
            idx, location=each_pos, sensor_resolution=meta_vars['sensor_resolution'], value=meta_vars['leader_value'],
            observation_radius=meta_vars['leader_obs_rad'], max_velocity=meta_vars['leader_max_velocity'], policy=None)
        for idx, each_pos in enumerate(experiment_config['leader_positions'])
    ]
    followers = [
        Follower(
            agent_id=idx, location=each_pos, sensor_resolution=meta_vars['sensor_resolution'],
            value=meta_vars['follower_value'], max_velocity=meta_vars['follower_max_velocity'],
            repulsion_radius=meta_vars['repulsion_rad'], repulsion_strength=meta_vars['repulsion_strength'],
            attraction_radius=meta_vars['attraction_rad'], attraction_strength=meta_vars['attraction_strength'])
        for idx, each_pos in enumerate(experiment_config['follower_positions'])
    ]
    pois = [
        Poi(idx, location=each_pos, sensor_resolution=meta_vars['sensor_resolution'], value=meta_vars['poi_value'],
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
    print(f'Using device: {list(agent_pops.values())[0][0]["network"].device()}')

    # initial fitness evaluation of all networks in population
    for pop_idx in range(meta_vars['population_size']):
        new_inds = {agent_name: policy_info[pop_idx] for agent_name, policy_info in agent_pops.items()}
        agent_rewards = rollout(env, new_inds, reward_func=reward_func, render=False)
        for agent_name, policy_info in agent_pops.items():
            policy_fitness = agent_rewards[agent_name]
            policy_info[pop_idx]['fitness'] = policy_fitness
    ########################################################

    start_time = time.time()
    best_solution = neuro_evolve(
        env, agent_pops, meta_vars['population_size'], meta_vars['n_gens'], meta_vars['sim_pop_size'],
        reward_func=reward_func, experiment_dir=experiment_dir
    )
    end_time = time.time()
    print(f'Time to train: {end_time - start_time}')

    rewards = rollout(env, best_solution, reward_func=reward_func)
    print(f'{rewards=}')
    return

def main(main_args):
    config_names = [
        # 'alpha',
        'atrium',
        # 'battery',
        # 'charlie',
        # 'echo'
    ]
    config_fns = [
        each_fn
        for each_fn in Path(project_properties.config_dir).rglob('*.yaml')
        if each_fn.stem in config_names
    ]
    print(f'{project_properties.config_dir=}')
    stat_runs = 2

    for each_fn in config_fns:
        print(f'{"=" * 80}')
        print(f'{each_fn}')

        exp_config = load_config(each_fn)

        for reward_key in reward_map.keys():
            now = datetime.now()
            experiment_id = f'{now.strftime("%Y_%m_%d_%H_%M")}'
            env_tag = f'{experiment_id}_{reward_key}_{each_fn.stem}'

            reward_path = Path(project_properties.cached_dir, 'experiments', env_tag)
            if not reward_path.exists():
                reward_path.mkdir(parents=True, exist_ok=True)

            for idx in range(0, stat_runs):
                exp_path = Path(reward_path, f'stat_run_{idx}')
                if not exp_path.exists():
                    exp_path.mkdir(parents=True, exist_ok=True)

                run_experiment(experiment_config=exp_config, experiment_dir=exp_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
