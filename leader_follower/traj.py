from pathlib import Path
from typing import Optional

import numpy as np
from pandas import DataFrame, read_csv
from matplotlib.pyplot import show
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from leader_follower import project_properties
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.agent import Leader

def save_trajectories(env: LeaderFollowerEnv):
    # print(env.leaders["leader_0"].state_history)

    labels = ['t','x', 'y', 'name', 'leader', 'poi', 'observed', 'Label']
    data = []

    agents = list(env.leaders.items()) + list(env.followers.items())
    for agent_name, agent in agents:
        for t, state in enumerate(agent.state_history):
            if type(agent) == Leader:
                label = "Leader " + agent_name[-1]
            else:
                label = "Follower"
            data.append([t, state[0], state[1], agent_name, type(agent)==Leader, False, False, label])

    for poi_name, poi in env.pois.items():
        for t in range(len(poi.observation_history)+1):
            if poi.observed:
                label = "Observed POI"
            else:
                label = "Unobserved POI"
            data.append([t, poi.state[0], poi.state[1], poi_name, False, True, poi.observed, label])

    traj_df = DataFrame(data=data, columns=labels)

    traj_dir = Path(project_properties.output_dir, 'trajs')
    if not traj_dir.exists():
        traj_dir.mkdir(parents=True, exist_ok=True)
    
    num_trajs = len(list(traj_dir.iterdir()))
    trajname = Path(traj_dir,f'trajectories_{num_trajs}.csv')

    traj_df.to_csv(trajname, index=False)

    # print(traj_df)

def load_trajectories(traj_num: Optional[int] = None):
    traj_dir = Path(project_properties.output_dir, 'trajs')
    if not traj_dir.exists():
        raise Exception("Could not load trajectories because trajectories folder was not found.")
    
    if traj_num is None:
        traj_num = len(list(traj_dir.iterdir())) - 1
    
    trajname = Path(traj_dir,f'trajectories_{traj_num}.csv')

    return read_csv(trajname)

def fake_error(vector):
    return (min(vector), max(vector))

def plot_trajectories(traj_df: DataFrame):
    sns.set_theme()

    custom_pallete = {"Follower": (0.5,0.7,0.7), "Unobserved POI": (1,0,0), "Observed POI":(0,1,0)}
    label_set = set(traj_df["Label"])
    leaders = [item for item in label_set if "Leader" in item]
    for count, leader_name in enumerate(leaders):
        custom_pallete[leader_name] = "C"+str(count%10)


    g=sns.lineplot(
        data=traj_df, 
        x="x", y="y", 
        hue="Label", units="name", 

        palette= custom_pallete, #{"Leader 0": "C0", "Leader 1": "C0", "Follower": "C1", "Unobserved POI": "k", "Observed POI":"C2"},
        marker="o", sort=False, estimator=None
    )
    g.set(ylim=(0,15), xlim=(0,15))
    g.set_xticks(list(range(16)))
    g.set_yticks(list(range(16)))

    plt.show()





        # p = so.Plot(traj_df, "x", "y", color="Label", marker="name" \
        #     ).add(so.Path(marker="o", pointsize=2, linewidth=0.75, fillcolor="w") \
        #         )

        
        # f, ax = plt.subplots()
        # # ax = f.add_axes([0.1, 0.1, 0.6, 0.75])
        # res = p.on(ax).plot()
        # ax.axis('equal')
        # ax.set_xlim([0,15])
        # ax.set_ylim([0,15])
        # ax.set_xticks(list(range(16)))
        # ax.set_yticks(list(range(16)))

        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width*0.8, box.height])

        # # plt.legend(loc="upper right")

        # # sns.move_legend(ax, "upper_left", bbox_to_anchor=(1,1))

        # res.show()
