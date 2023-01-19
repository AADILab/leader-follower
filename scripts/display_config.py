"""
@title

@description

"""
import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from leader_follower import project_properties
from leader_follower.agent import Leader, Follower, Poi
from leader_follower.leader_follower_env import LeaderFollowerEnv
from leader_follower.learn.neural_network import NeuralNetwork
from leader_follower.utils import load_config


def generate_plot(config_path: Path):
    render_mode = 'rgb_array'
    experiment_config = load_config(str(config_path))
    config_name = config_path.stem

    # agent_id, policy_population: list[NeuralNetwork], location, velocity, sensor_resolution, observation_radius, value
    leaders = [
        Leader(idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, value=1,
               observation_radius=0, policy=NeuralNetwork(n_inputs=8, n_hidden=2, n_outputs=2))
        for idx, each_pos in enumerate(experiment_config['leader_positions'])
    ]
    # agent_id, update_rule, location, velocity, sensor_resolution, observation_radius, value
    followers = [
        Follower(agent_id=idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, value=1,
                 repulsion_radius=0, repulsion_strength=2,
                 attraction_radius=0, attraction_strength=1)
        for idx, each_pos in enumerate(experiment_config['follower_positions'])
    ]
    #  agent_id, location, velocity, sensor_resolution, observation_radius, value, coupling
    pois = [
        Poi(idx, location=each_pos, velocity=(0, 0), sensor_resolution=4, value=1,
            observation_radius=0, coupling=1)
        for idx, each_pos in enumerate(experiment_config['poi_positions'])
    ]

    # leaders: list[Leader], followers: list[Follower], pois: list[Poi], max_steps, delta_time=1, render_mode=None
    env = LeaderFollowerEnv(
        leaders=leaders, followers=followers, pois=pois, max_steps=100, render_mode=render_mode, delta_time=0
    )
    frame = env.render()

    config_name = config_name.split(".")[0]
    fig_name = f'{config_name}'

    fg_color = 'white'
    bg_color = 'black'
    fig, axes = plt.subplots(1, 1)

    img = axes.imshow(frame)
    # set visibility of x-axis as False
    xax = axes.get_xaxis()
    xax = xax.set_visible(False)

    # set visibility of y-axis as False
    yax = axes.get_yaxis()
    yax = yax.set_visible(False)

    axes.set_title(fig_name, color=fg_color)
    axes.patch.set_facecolor(bg_color)

    # set tick and ticklabel color
    img.axes.tick_params(color=fg_color, labelcolor=fg_color)

    # set imshow outline
    for spine in img.axes.spines.values():
        spine.set_edgecolor(fg_color)
    fig.patch.set_facecolor(bg_color)
    plt.tight_layout()

    save_name = Path(project_properties.doc_dir, 'configs', f'{fig_name}.png')
    if not save_name.parent.exists():
        save_name.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(save_name))
    plt.close()
    return
def main(main_args):
    delta_time = 1

    leader_obs_rad = 5
    repulsion_rad = 2
    attraction_rad = 5

    render_delay = 5
    ############################################################
    config_paths = Path(project_properties.config_dir).glob('*.yaml')
    for each_path in config_paths:
        if each_path.stem == 'meta_params':
            continue

        # config_fn = Path(project_properties.test_dir, 'configs', config_name)
        generate_plot(each_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
