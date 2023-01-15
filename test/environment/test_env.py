"""
@title

@description

"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
from pettingzoo.test import parallel_api_test

from leader_follower import project_properties
from leader_follower.agent import Poi, Follower, Leader
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork
from leader_follower.utils import load_config


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

    config_fn = Path(project_properties.config_dir, 'test.yaml')
    experiment_config = load_config(str(config_fn))

    # agent_id, policy_population: list[NeuralNetwork], location, velocity, sensor_resolution, observation_radius, value
    leaders = [
        Leader(idx, location=(1, 1), velocity=(0, 0), sensor_resolution=4, observation_radius=1, value=1,
               policy_population=[NeuralNetwork(8, 2, 2)])
        for idx, each_pos in enumerate(experiment_config['leader_positions'])
    ]
    # agent_id, update_rule, location, velocity, sensor_resolution, observation_radius, value
    followers = [
        Follower(agent_id=idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, observation_radius=1, value=1,
                 repulsion_radius=0.25, attraction_radius=2)
        for idx, each_pos in enumerate(experiment_config['follower_positions'])
    ]
    #  agent_id, location, velocity, sensor_resolution, observation_radius, value, coupling
    pois = [
        Poi(idx, location=(1, 9), velocity=(0, 0), sensor_resolution=4, observation_radius=1, value=1, coupling=1)
        for idx, each_pos in enumerate(experiment_config['poi_positions'])
    ]

    # leaders: list[Leader], followers: list[Follower], pois: list[Poi], max_steps, delta_time=1, render_mode=None
    env = LeaderFollowerEnv(
        leaders=leaders, followers=followers, pois=pois, max_steps=100, render_mode=render_mode, delta_time=delta_time
    )

    test_observations(env)
    test_actions(env)
    test_render(env)
    #
    test_step(env, render=None)
    test_step(env, render='rgb_array')
    #
    test_random(env, render=None)
    test_random(env, render='rgb_array')
    #
    test_api(env)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
