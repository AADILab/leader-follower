from pathlib import Path
from typing import Optional

from pandas import DataFrame, read_csv

from leader_follower import project_properties
from leader_follower.leader_follower_env import LeaderFollowerEnv

def save_trajectories(env: LeaderFollowerEnv):
    # print(env.leaders["leader_0"].state_history)
    traj_dict = {
        leader_name: {
            "state_history": leader.state_history,
            "action_history": leader.action_history,
            "observation_history": leader.observation_history
        } for leader_name, leader in env.leaders.items()
    } | {
        follower_name: {
            "state_history": follower.state_history,
            "action_history": follower.action_history,
            "observation_history": follower.observation_history
        } for follower_name, follower in env.followers.items()
    }
    traj_df = DataFrame.from_dict(traj_dict)

    traj_dir = Path(project_properties.output_dir, 'trajs')
    if not traj_dir.exists():
        traj_dir.mkdir(parents=True, exist_ok=True)
    
    num_trajs = len(list(traj_dir.iterdir()))
    trajname = Path(traj_dir,f'trajectories_{num_trajs}.csv')

    traj_df.to_csv(trajname)

def load_trajectories(traj_num: Optional[int] = None):
    traj_dir = Path(project_properties.output_dir, 'trajs')
    if not traj_dir.exists():
        raise Exception("Could not load trajectories because trajectories folder was not found.")
    
    if traj_num is None:
        traj_num = len(list(traj_dir.iterdir())) - 1
    
    trajname = Path(traj_dir,f'trajectories_{traj_num}.csv')

    return read_csv(trajname)
