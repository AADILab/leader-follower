"""
@title

@description

"""
import argparse
import csv
from pathlib import Path

import numpy as np

from leader_follower import project_properties
from leader_follower.agent import Leader, Follower, Poi
from leader_follower.leader_follower_env import LeaderFollowerEnv


def actions_from_states(states):
    prev_loc = (states[0]['x'], states[0]['y'])
    actions = []
    for row in states[1:]:
        each_loc = (row['x'], row['y'])
        act = (each_loc[0] - prev_loc[0], each_loc[1] - prev_loc[1])
        prev_loc = each_loc

        act = np.array(act)
        actions.append(act)
    return actions

def reconstruct_leader(traj, a_id):
    obs_rad = 100
    value = 1

    prev_loc = (traj[0]['x'], traj[0]['y'])
    new_leader = Leader(
        agent_id=a_id, location=prev_loc, sensor_resolution=4, value=value, observation_radius=obs_rad, policy=None
    )
    actions = actions_from_states(traj)
    return new_leader, actions

def reconstruct_follower(traj, a_id):
    value = 1
    repulsion_rad = 0.5
    attraction_rad = 1

    prev_loc = (traj[0]['x'], traj[0]['y'])
    new_follower = Follower(
        agent_id=a_id, location=prev_loc, sensor_resolution=4, value=value,
        repulsion_radius=repulsion_rad, repulsion_strength=1,
        attraction_radius=attraction_rad, attraction_strength=1
    )
    actions = actions_from_states(traj)
    return new_follower, actions

def reconstruct_poi(traj, a_id):
    obs_rad = 1
    value = 0
    coupling = 3

    prev_loc = (traj[0]['x'], traj[0]['y'])
    new_poi = Poi(
        agent_id=a_id, location=prev_loc, sensor_resolution=4, value=value,
        observation_radius=obs_rad, coupling=coupling
    )
    actions = actions_from_states(traj)
    return new_poi, actions

def reconstruct_env(agent_trajs):
    reconstruct_mapping = {
        'leader': reconstruct_leader,
        'follower': reconstruct_follower,
        'poi': reconstruct_poi,
    }
    agents = {
        'leader': [],
        'follower': [],
        'poi': [],
    }
    agent_actions = {}
    for agent_name, agent_traj in agent_trajs.items():
        agent_type = agent_name.split('_')[0]
        agent_id = agent_name.split('_')[1]
        rec_func = reconstruct_mapping[agent_type]
        print(agent_name)
        agent, actions = rec_func(agent_traj, agent_id)
        agents[agent_type].append(agent)
        agent_actions[agent_name] = actions
    first_actions = list(agent_actions.values())[0]
    max_steps = len(first_actions)
    env = LeaderFollowerEnv(
        leaders=agents['leader'], followers=agents['follower'], pois=agents['poi'],
        max_steps=max_steps, delta_time=1, render_mode=None
    )

    # rollout of stored actions to determine observations of agents
    episode_actions = [dict(zip(agent_actions, t)) for t in zip(*agent_actions.values())]
    for each_action in episode_actions:
        print(each_action)
        observations, rewards, terminations, truncs, infos = env.step(each_action)

    # store actions in each agent's action_history
    for agent_name, actions in agent_actions.items():
        agent = env.agent_mapping[agent_name]
        agent.action_history = actions
    return env

def validate_states(env, agent_name, true_traj):
    valid = True
    agent = env.agent_mapping[agent_name]
    for truth_entry in true_traj:
        time_step = truth_entry['t']
        true_loc = (truth_entry['x'], truth_entry['y'])
        agent_state = agent.state_history[time_step]
        if agent_state != true_loc:
            valid = False
    return valid

def validate_observed(env, agent_name, true_traj):
    valid = True
    agent = env.agent_mapping[agent_name]
    for truth_entry in true_traj:
        time_step = truth_entry['t']
        true_observed = truth_entry['observed'] == 'True'

        # todo look at observation history to determine if the agent was observed at this time step
        # todo check csv file for if it saves it at each time step or only at end
        # agent_observation = agent.observation_history[time_step]
        # agent_observed = len(agent_observation) >= agent.coupling
        agent_observed = agent.observed
        if agent_observed != true_observed:
            valid = False
    return valid

def validate_reconstruction(env, agent_trajs):
    # validate against stored values to make sure observed is being correctly determined
    agent_valids = {}
    for agent_name, traj in agent_trajs.items():
        agent_type = agent_name.split('_')[0]
        valid = validate_states(env, agent_name, traj)
        if agent_type == 'poi':
            valid_poi = validate_observed(env, agent_name, traj)
            valid = valid and valid_poi
        agent_valids[agent_name] = valid
    return agent_valids

def parse_traj_csv(traj_fname):
    agent_trajs = {}
    with open(traj_fname, 'r') as traj_file:
        reader = csv.DictReader(traj_file)
        for row in reader:
            agent_name = row['name']
            if agent_name not in agent_trajs:
                agent_trajs[agent_name] = []
            entry = {
                't': int(row['t']), 'x': float(row['x']), 'y': float(row['y']),
                'type': agent_name.split('_')[0], 'observed': row['observed']
            }
            agent_trajs[agent_name].append(entry)
    for each_agent, trajs in agent_trajs.items():
        trajs.sort(key=lambda x: x['t'])
    return agent_trajs

def main(main_args):
    # todo rewrite to read from experiments in cached directory
    traj_fname = Path(project_properties.project_path, 'bugs', 'trajectories_54.csv')
    agent_trajs = parse_traj_csv(traj_fname)
    env = reconstruct_env(agent_trajs)

    print(f'validating environment reconstruction')
    valid = validate_reconstruction(env, agent_trajs)
    for name, isvalid in valid.items():
        print(f'{name=} | {isvalid=}')

    rewards = env.calc_diff_rewards()
    print(f'{rewards=}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
