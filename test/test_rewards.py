"""
@title

@description

"""
import argparse
from functools import partial

import numpy as np

from leader_follower.agent import Poi, Leader, Follower
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork

def leader_follower_test(leaders, leader_actions, followers, follower_actions, pois, poi_actions, reward_func):
    """
        Run a single test of a set of leaders and followers to capture a poi.

        This should add leaders, followers, and pois to a LeaderFollowerEnv, rollout the agents
        using predefined actions, and calculate the reward.

        By creating the env in this function, each instance is separate
        and does not need to care about reinitialization.

        :return:
        """
    render_mode = 'rgb_array'
    delta_time = 1
    num_steps = len(leader_actions)

    env = LeaderFollowerEnv(
        leaders=leaders, followers=followers, pois=pois, max_steps=100, render_mode=render_mode, delta_time=delta_time
    )

    for step_idx in range(0, num_steps):
        step_actions = leader_actions[step_idx]
        step_actions.update(follower_actions[step_idx])
        step_actions.update(poi_actions[step_idx])

        observations, rewards, terminations, truncs, infos = env.step(step_actions)
        print(f'{step_idx=}')
        for agent_name, obs in observations.items():
            agent = env.agent_mapping[agent_name]
            a_reward = rewards[agent_name]
            a_done = terminations[agent_name]
            print(f'\t{agent=} | {obs=} | {a_reward=} | {a_done=}')
    rewards = reward_func(env)
    print(f'{rewards=}')
    return rewards


def test_leaders_followers(
        leader_info, follower_info,
        poi_positions, reward_func, tag, max_value=4
):
    assert len(leader_info) == len(follower_info)
    assert len(leader_info) == len(poi_positions)

    print(f'{tag}')
    sensor_resolution = 4
    leader_obs_rad = 5

    repulsion_rad = 0.5
    repulsion_strength = 5
    attraction_rad = 2
    attraction_strength = 1

    poi_obs_rad = 1.5
    poi_value = 0
    poi_coupling = 3
    rollout_len = 5

    max_value = poi_coupling + 1 if max_value <= 0 else max_value
    for set_idx, info in enumerate(zip(leader_info, follower_info, poi_positions)):

        leader_info_set = info[0]
        leader_pos_set = leader_info_set[0]
        leader_act = leader_info_set[1]

        follower_info_set = info[1]
        follower_pos_set = follower_info_set[0]
        follower_act = follower_info_set[1]

        poi_pos_set = info[2]

        for each_val in range(1, max_value):
            leaders = [
                Leader(
                    idx, location=each_pos, sensor_resolution=sensor_resolution, value=each_val,
                    observation_radius=leader_obs_rad, policy=NeuralNetwork(n_inputs=8, n_hidden=2, n_outputs=2)
                )
                for idx, each_pos in enumerate(leader_pos_set)
            ]
            leader_actions = [
                {agent.name: leader_act for agent in leaders}
                for _ in range(0, rollout_len)
            ]

            followers = [
                Follower(agent_id=idx, location=each_pos, sensor_resolution=sensor_resolution, value=each_val,
                         repulsion_radius=repulsion_rad, repulsion_strength=repulsion_strength,
                         attraction_radius=attraction_rad, attraction_strength=attraction_strength)
                for idx, each_pos in enumerate(follower_pos_set)
            ]
            follower_actions = [
                {agent.name: follower_act for agent in followers}
                for _ in range(0, rollout_len)
            ]

            pois = [
                Poi(
                    idx, location=each_pos, sensor_resolution=sensor_resolution, value=poi_value,
                    observation_radius=poi_obs_rad, coupling=poi_coupling
                )
                for idx, each_pos in enumerate(poi_pos_set)
            ]
            poi_actions = [
                {agent.name: (0, 0) for agent in pois}
                for _ in range(0, rollout_len)
            ]

            leader_follower_test(
                leaders=leaders, leader_actions=leader_actions,
                followers=followers, follower_actions=follower_actions,
                pois=pois, poi_actions=poi_actions,
                reward_func=reward_func
            )
    return

def main(main_args):
    # action is (dx, dy)
    null_action = np.array((0, 0))
    up_action = np.array((0, 1))
    down_action = np.array((0, -1))
    left_action = np.array((-1, 0))
    right_action = np.array((1, 0))

    rewards = {
        'global': LeaderFollowerEnv.calc_global,

        'diff': partial(LeaderFollowerEnv.calc_diff_rewards, **{'remove_followers': False}),
        'diff_lf': partial(LeaderFollowerEnv.calc_diff_rewards, **{'remove_followers': True}),

        'dpp': partial(LeaderFollowerEnv.calc_dpp, **{'remove_followers': False}),
        'dpp_lf': partial(LeaderFollowerEnv.calc_dpp, **{'remove_followers': True})
    }

    #########################################################################

    leader_infos = [
        [[(0, -2)], up_action],
        [[(0, -2), (0, -2.5)], up_action],
        [[(0, -2), (0, -2.5), (0, -3)], up_action],

        [[(3, 0)], left_action],
        [[(3, 0), (3.5, 0)], left_action],
        [[(3, 0), (3.5, 0), (4, 0)], left_action],

        [[(-3, 0)], right_action],
        [[(-3, 0), (-3.5, 0)], right_action],
        [[(-3, 0), (-3.5, 0), (-4, 0)],  right_action],

        # start tests with including followers
        [[(0, -2)], up_action],
        [[(0, -2), (0, -2.5)], up_action],
        [[(0, -2), (0, -2.5), (0, -3)], up_action],

        [[(3, 0)], left_action],
        [[(3, 0), (3.5, 0)], left_action],
        [[(3, 0), (3.5, 0), (4, 0)], left_action],

        [[(-3, 0)], right_action],
        [[(-3, 0), (-3.5, 0)], right_action],
        [[(-3, 0), (-3.5, 0), (-4, 0)], right_action],
    ]

    # todo add follower positions for designed tests
    follower_infos = [
        [[], up_action],
        [[], up_action],
        [[], up_action],

        [[], left_action],
        [[], left_action],
        [[], left_action],

        [[], right_action],
        [[], right_action],
        [[], right_action],

        # start tests with including followers
        [[], up_action],
        [[], up_action],
        [[], up_action],

        [[], left_action],
        [[], left_action],
        [[], left_action],

        [[], right_action],
        [[], right_action],
        [[], right_action],
    ]

    poi_positions = [
        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],

        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],

        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],

        # start tests with including followers
        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],

        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],

        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],
        [(0, 1), (1, 0), (-1, 0)],
    ]

    #########################################################################

    for rew_name, each_rew in rewards.items():
        test_leaders_followers(
            leader_info=leader_infos, follower_info=follower_infos, poi_positions=poi_positions,
            reward_func=each_rew, tag=rew_name
        )
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
