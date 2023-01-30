"""
@title

@description

"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from leader_follower import project_properties
from leader_follower.agent import Leader, Follower, Poi
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.cceaV2 import neuro_evolve, rollout
from leader_follower.learn.neural_network import load_pytorch_model
from leader_follower.utils import load_config
from scripts.learnV2 import reward_map


def restart_stat_run(stat_run_dir):
    # load meta file of parameters
    meta_fname = Path(stat_run_dir, 'meta_vars.json')
    with open(meta_fname, 'r') as jfile:
        meta_vars = json.load(jfile)

    gen_dirs = list(stat_run_dir.glob('gen_*'))
    gen_dirs = sorted(gen_dirs, key=lambda x: int(x.stem.split('_')[1]))
    last_gen = gen_dirs[-1]

    leader_dirs = list(last_gen.glob('leader_*_networks'))
    agent_pops = {}
    for each_dir in leader_dirs:
        leader_name = each_dir.stem.split('_')[:-1]
        leader_name = '_'.join(leader_name)
        agent_pops[leader_name] = []

        # load in leader policies

        network_fnames = list(each_dir.glob('*_model_*.pt'))
        for each_fname in network_fnames:
            network = load_pytorch_model(each_fname)
            a_network = {
                'network': network,
                'fitness': None
            }
            agent_pops[leader_name].append(a_network)

    fitnesses_fname = Path(last_gen, 'fitnesses.csv')
    fitnesses_df = pd.read_csv(fitnesses_fname)
    # correlate policies to fitnesses
    for agent_name, policies in agent_pops.items():
        agent_row = fitnesses_df.loc[fitnesses_df['agent_name'] == agent_name]
        fitness_values = agent_row.values[0]
        for idx, each_policy in enumerate(policies):
            fitness = fitness_values[idx + 1]
            each_policy['fitness'] = fitness

    # load positions of agents
    position_fname = Path(project_properties.config_dir, f"{meta_vars['config_name']}.yaml")
    experiment_config = load_config(position_fname)

    # create agents and environment
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
    reward_func = reward_map[meta_vars['reward_key']]

    # start neuro_evolve from specified generation
    print(f'Restarting experiment: {meta_vars["reward_key"]} | {meta_vars["config_name"]}')
    last_gen_idx = len(gen_dirs)
    start_time = time.time()
    best_solution = neuro_evolve(
        env, agent_pops, meta_vars['population_size'], meta_vars['n_gens'], meta_vars['sim_pop_size'],
        reward_func=reward_func, experiment_dir=meta_vars['experiment_dir'], starting_gen=last_gen_idx
    )
    end_time = time.time()
    print(f'Time to train: {end_time - start_time}')

    rewards = rollout(env, best_solution, reward_func=reward_func)
    print(f'{rewards=}')
    return

def main(main_args):
    stat_run_dirs = list(Path(project_properties.cached_dir, 'experiments').rglob('*/stat_run*'))
    for each_dir in stat_run_dirs:
        restart_stat_run(each_dir)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
