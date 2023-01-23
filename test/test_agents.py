"""
@title

@description

"""
import argparse

from leader_follower.agent import Follower, Leader, Poi
from leader_follower.leader_follower_env import LeaderFollowerEnv


def test_leader():
    base_loc = (5, 5)
    base_leader = Leader(0, location=base_loc, sensor_resolution=4, observation_radius=5, value=1, policy=None)
    surround_locs = [(5, 4), (5, 6), (4, 5), (6, 5), (4, 4), (4, 6), (6, 4), (6, 6)]

    for each_loc in surround_locs:
        each_leader = Leader(0, location=each_loc, sensor_resolution=4, observation_radius=5, value=1, policy=None)
        leads = [base_leader, each_leader]
        env = LeaderFollowerEnv(leaders=leads, followers=[], pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print(f'{base_loc=} | {each_loc=} | {obs=} | {acts=}')
    return


def test_follower():
    base_loc = (5, 5)
    base_follower = Follower(
        0, location=base_loc, sensor_resolution=4, value=1,
        attraction_radius=1, attraction_strength=1, repulsion_radius=1, repulsion_strength=1
    )
    surround_locs = [(5, 4), (5, 6), (4, 5), (6, 5), (4, 4), (4, 6), (6, 4), (6, 6)]

    for each_loc in surround_locs:
        each_follower = Follower(0, location=each_loc,  sensor_resolution=4, value=1,
                                 attraction_radius=1, attraction_strength=1,
                                 repulsion_radius=1, repulsion_strength=1
                                 )
        followers = [base_follower, each_follower]
        env = LeaderFollowerEnv(leaders=[], followers=followers, pois=[], max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print(f'{base_loc=} | {each_loc=} | {obs=} | {acts=}')
    return


def test_poi():
    base_loc = (5, 5)
    base_poi = Poi(
        0, location=base_loc, sensor_resolution=4, observation_radius=1, value=1,coupling=1)
    surround_locs = [(5, 4), (5, 6), (4, 5), (6, 5), (4, 4), (4, 6), (6, 4), (6, 6)]

    for each_loc in surround_locs:
        each_poi = Poi(0, location=each_loc,  sensor_resolution=4, observation_radius=1, value=1, coupling=1)
        pois = [base_poi, each_poi]
        env = LeaderFollowerEnv(leaders=[], followers=[], pois=pois, max_steps=100, render_mode=None, delta_time=1)
        obs = env.get_observations()
        acts = env.get_actions()
        print(f'{base_loc=} | {each_loc=} | {obs=} | {acts=}')
    return


def main(main_args):
    test_leader()
    test_follower()
    test_poi()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
