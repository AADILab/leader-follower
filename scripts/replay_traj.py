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
    prev_loc = (traj[0]['x'], traj[0]['y'])
    new_leader = Leader(
        agent_id=a_id, location=prev_loc, sensor_resolution=4, value=1, observation_radius=1, policy=None
    )
    actions = actions_from_states(traj)
    return new_leader, actions
def reconstruct_follower(traj, a_id):
    prev_loc = (traj[0]['x'], traj[0]['y'])
    new_follower = Follower(
        agent_id=a_id, location=prev_loc, sensor_resolution=4, value=1,
        repulsion_radius=5, repulsion_strength=1,
        attraction_radius=5, attraction_strength=1
    )
    actions = actions_from_states(traj)
    return new_follower, actions

def reconstruct_poi(traj, a_id):
    prev_loc = (traj[0]['x'], traj[0]['y'])
    # todo determine coupling requirement
    new_poi = Poi(agent_id=a_id, location=prev_loc, sensor_resolution=4, value=0, observation_radius=1, coupling=1)
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
    valid = validate_reconstruction(env, agent_trajs)
    return env, valid

def validate_reconstruction(env, agent_trajs):
    # todo validate against stored values to make sure observed is being correctly determined
    return False

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
    traj_fname = Path(project_properties.project_path, 'bugs', 'trajectories_54.csv')
    agent_trajs = parse_traj_csv(traj_fname)
    env = reconstruct_env(agent_trajs)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
