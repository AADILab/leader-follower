"""
@title

@description

"""
import argparse
import time

import matplotlib.pyplot as plt
from pettingzoo.test import parallel_api_test

from leader_follower.agent import Poi, Follower, Leader, FollowerRule
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork


def display_finale_agents(env):
    print(f'Remaining agents: {len(env.agents)}')
    for agent_name in env.agents:
        agent = env.agent_mapping[agent_name]
        print(f'{agent_name=}: {agent.location=}')
    print(f'Completed agents: {len(env.completed_agents)}')
    for agent_name, agent_reward in env.completed_agents.items():
        agent = env.agent_mapping[agent_name]
        print(f'{agent_name=} | {agent_reward=} | {agent.location=}')
    return


def test_observations(env):
    print(f'=' * 80)
    env.reset()
    print(f'Running observation tests')
    obs_space = env.observation_space(env.agents[0])
    print(f'{obs_space=}')

    for agent_name in env.agents:
        each_obs = env.observation_space(agent_name)
        print(f'{agent_name}: {each_obs}')
    all_obs = env.get_observations()
    for agent_name, each_obs in all_obs.items():
        print(f'{agent_name}: {each_obs}')
    print(f'=' * 80)
    return


def test_actions(env):
    print(f'=' * 80)
    env.reset()
    print(f'Running action tests')
    act_space = env.action_space(env.agents[0])
    print(f'{act_space=}')

    for each_agent in env.agents:
        each_act = env.action_space(each_agent)
        print(f'{each_agent}: {each_act}')

    all_obs = env.get_observations()
    for agent_name, obs in all_obs.items():
        agent = env.agent_mapping[agent_name]
        action = agent.get_action(obs)
        print(f'{agent_name=}: {obs=} | {action=}')
    print(f'=' * 80)
    return


def test_render(env):
    print(f'=' * 80)
    env.reset()
    print(f'Running render tests')
    mode = env.render_mode
    print(f'{mode=}')

    frame = env.render(mode='rgb_array')
    env.close()

    plt.imshow(frame)
    plt.show()
    plt.close()

    print(f'=' * 80)
    return


def test_step(env, render):
    render_delay = 0.1

    # action is (vx, vy)
    forward_action = (0, 1)
    backwards_action = (0, -1)
    left_action = (-1, 0)
    right_action = (1, 0)

    tests = [
        {agent: forward_action for agent in env.agents},
        {agent: backwards_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: left_action for agent in env.agents},

        {agent: forward_action for agent in env.agents},
        {agent: forward_action for agent in env.agents},
        {agent: forward_action for agent in env.agents},
        {agent: forward_action for agent in env.agents},

        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
    ]

    print(f'=' * 80)
    env.reset()
    print(f'Running step tests')
    obs_space = env.observation_space(env.agents[0])
    act_space = env.action_space(env.agents[0])
    print(f'{obs_space=}')
    print(f'{act_space=}')

    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render:
            frame = env.render(render)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)

    # reset and do it again
    env.reset()
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render:
            frame = env.render(render)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)
    display_finale_agents(env)
    print(f'=' * 80)
    return


def test_random(env, render):
    render_delay = 0.1
    counter = 0
    done = False
    print(f'=' * 80)
    env.reset()
    print(f'Running random step tests')

    # noinspection DuplicatedCode
    init_observations = env.reset()
    print(f'{init_observations=}')
    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0:
            print(f'{counter=}')
        if render:
            frame = env.render(render)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)

    # reset and do it again
    # noinspection DuplicatedCode
    init_observations = env.reset()
    print(f'{init_observations=}')
    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0:
            print(f'{counter=}')
        if render:
            frame = env.render(render)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)

    display_finale_agents(env)
    print(f'=' * 80)
    return


def test_api(env):
    print(f'=' * 80)
    print(f'Running parallel api tests')
    result = parallel_api_test(env, num_cycles=50)
    print(f'{result=}')
    display_finale_agents(env)
    print(f'=' * 80)
    return


def main(main_args):
    max_steps = 100
    render_mode = 'rgb_array'
    delta_time = .1
    obs_rad = 2
    velocity_range = (-1, 1)
    state_res = 4

    # n_inputs, n_outputs, n_hidden=2, network_func=linear_layer, name=None
    # neural_network = NeuralNetwork(n_inputs=8, n_outputs=2, n_hidden=2)

    # agent_id: int, location, velocity, sensor_resolution, velocity_range, observation_radius,
    # policy_population: list[NeuralNetwork]
    leaders = [
        Leader(0, (1, 1), (0, 0), state_res, velocity_range, obs_rad,
               [NeuralNetwork(8, 2, 2)]),
        Leader(1, (3.5, 1), (0, 0), state_res, velocity_range, obs_rad,
               [NeuralNetwork(8, 2, 2)]),
        Leader(2, (6, 1), (0, 0), state_res, velocity_range, obs_rad,
               [NeuralNetwork(8, 2, 2)]),
        Leader(3, (8.5, 1), (0, 0), state_res, velocity_range, obs_rad,
               [NeuralNetwork(8, 2, 2)]),
    ]
    # agent_id: int, location, velocity, sensor_resolution, velocity_range, observation_radius,
    # repulsion, attraction, alignment
    followers = [
        Follower(0, (0.5, 0), (0, 0), state_res, velocity_range, obs_rad,
                 FollowerRule(1, 1), FollowerRule(2, 1), FollowerRule(3, 1)),
        Follower(1, (1.5, 0), (0, 0), state_res, velocity_range, obs_rad,
                 FollowerRule(1, 1), FollowerRule(2, 1), FollowerRule(3, 1)),

        Follower(2, (3, 0), (0, 0), state_res, velocity_range, obs_rad,
                 FollowerRule(1, 1), FollowerRule(2, 1), FollowerRule(3, 1)),
        Follower(3, (4, 0), (0, 0), state_res, velocity_range, obs_rad,
                 FollowerRule(1, 1), FollowerRule(2, 1), FollowerRule(3, 1)),

        Follower(4, (5.5, 0), (0, 0), state_res, velocity_range, obs_rad,
                 FollowerRule(1, 1), FollowerRule(2, 1), FollowerRule(3, 1)),
        Follower(5, (6.5, 0), (0, 0), state_res, velocity_range, obs_rad,
                 FollowerRule(1, 1), FollowerRule(2, 1), FollowerRule(3, 1)),

        Follower(6, (8, 0), (0, 0), state_res, velocity_range, obs_rad,
                 FollowerRule(1, 1), FollowerRule(2, 1), FollowerRule(3, 1)),
        Follower(7, (9, 0), (0, 0), state_res, velocity_range, obs_rad,
                 FollowerRule(1, 1), FollowerRule(2, 1), FollowerRule(3, 1)),
    ]
    #  agent_id: int, location, velocity, sensor_resolution, velocity_range, observation_radius, value, coupling
    pois = [
        Poi(0, (1, 9), (0, 0), state_res, (0, 0), obs_rad, 1, 2),
        Poi(1, (3.5, 9), (0, 0), state_res, (0, 0), obs_rad, 1, 2),
        Poi(2, (6, 9), (0, 0), state_res, (0, 0), obs_rad, 1, 2),
        Poi(3, (8.5, 9), (0, 0), state_res, (0, 0), obs_rad, 1, 2),
    ]

    # pois: list[Poi], leaders: list[Leader], followers: list[Follower], max_steps: int
    env = LeaderFollowerEnv(
        leaders, followers, pois, max_steps=max_steps, render_mode=render_mode, delta_time=delta_time
    )

    # test_observations(env)
    # test_actions(env)
    # test_render(env)
    # test_step(env, render=None)
    test_step(env, render='rgb_array')
    # test_random(env, render=None)
    # test_random(env, render='rgb_array')
    # test_api(env)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
