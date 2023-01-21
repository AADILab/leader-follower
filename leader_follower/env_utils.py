import pickle
from pathlib import Path

from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower import project_properties

def save_env(env: LeaderFollowerEnv):
    env_dir = Path(project_properties.output_dir, 'envs')
    if not env_dir.exists():
        env_dir.mkdir(parents=True, exist_ok=True)
    num_envs = len(list(env_dir.iterdir()))
    envname = Path(env_dir,f'env_{num_envs}.pkl')

    pickle.dump(env, open(envname, "wb"))


def load_env():
    env_dir = Path(project_properties.output_dir, 'envs')
    if not env_dir.exists():
        env_dir.mkdir(parents=True, exist_ok=True)
    num_envs = len(list(env_dir.iterdir()))
    envname = Path(env_dir,f'env_{num_envs-1}.pkl')

    return pickle.load(open(envname, "rb"))
