"""
@title

@description

"""
import argparse
import time
from pathlib import Path

from leader_follower import project_properties
from leader_follower.agent import Leader, Follower, Poi
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.cceaV2 import neuro_evolve, rollout, plot_fitnesses
from leader_follower.learn.rewards import calc_global, calc_diff_rewards
from leader_follower.utils import load_config


def run_experiment(experiment_config, meta_config):
    leader_obs_rad = 5
    repulsion_rad = 0.5
    attraction_rad = 5

    # agent_id, policy_population: list[NeuralNetwork], location, velocity, sensor_resolution, observation_radius, value
    leaders = [
        Leader(idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, value=1,
               observation_radius=leader_obs_rad, policy=None)
        for idx, each_pos in enumerate(experiment_config['leader_positions'])
    ]
    # agent_id, update_rule, location, velocity, sensor_resolution, observation_radius, value
    followers = [
        Follower(agent_id=idx, location=each_pos, velocity=(0, 0),sensor_resolution=4, value=1,
                 repulsion_radius=repulsion_rad, repulsion_strength=2,
                 attraction_radius=attraction_rad, attraction_strength=1)
        for idx, each_pos in enumerate(experiment_config['follower_positions'])
    ]
    #  agent_id, location, velocity, sensor_resolution, observation_radius, value, coupling
    pois = [
        Poi(idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, value=1, observation_radius=leader_obs_rad,
            coupling=1)
        for idx, each_pos in enumerate(experiment_config['poi_positions'])
    ]

    # leaders: list[Leader], followers: list[Follower], pois: list[Poi], max_steps, delta_time=1, render_mode=None
    env = LeaderFollowerEnv(
        leaders=leaders, followers=followers, pois=pois, max_steps=meta_config['episode_length']
    )

    reward_map = {
        'global': calc_global,
        'difference': calc_diff_rewards,
        # 'dpp': calc_dpp
    }

    reward_func = reward_map['difference']
    n_hidden = 2
    sim_subpop_size = 15
    subpop_size = 25
    n_gens = 5000

    start_time = time.time()
    # env, n_hidden, population_size, n_gens, sim_pop_size, reward_func
    best_solution, max_fits, avg_fits = neuro_evolve(
        env, n_hidden, subpop_size, n_gens, sim_subpop_size, reward_func=reward_func
    )
    end_time = time.time()

    rewards = rollout(env, best_solution, reward_func=reward_func)
    print(f'{rewards=}')
    plot_fitnesses(avg_fitnesses=avg_fits, max_fitnesses=max_fits)
    # gw.plot_agent_trajectories()
    return

def main(main_args):
    config_names = [
        'alpha', 'charlie', 'echo'
    ]
    config_fns = [
        each_fn
        for each_fn in Path(project_properties.config_dir).rglob('*.yaml')
        if each_fn.stem in config_names
    ]
    config_name = Path(project_properties.config_dir, 'meta_params.yaml')
    meta_params = load_config(config_name)
    stat_runs = 5

    # subpop_size = 50
    # n_gens = 5000
    # meta_params['sub_population_size'] = subpop_size
    # meta_params['num_generations'] = n_gens
    for each_fn in config_fns:
        print(f'{"=" * 80}')
        print(f'{each_fn}')

        exp_config = load_config(each_fn)
        for idx in range(0, stat_runs):
            run_experiment(experiment_config=exp_config, meta_config=meta_params)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
