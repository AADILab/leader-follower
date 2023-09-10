from time import time, sleep
from sys import exit
from typing import List, Optional
import math

import seaborn as sns
import seaborn.objects as so
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import pygame
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from lib.boids_env import BoidsEnv
from lib.network_lib import createNNfromWeights
from lib.ccea_lib import computeAction
from lib.file_helper import getLatestTrialName, loadTrial, loadConfig

PLOT_BEST_SCORES = False

PLOT_AVERAGE_SCORES = False

PLOT_FINAL_EVALUATION_TEAMS = False

PLAY_ENV = True

PLOT_TRAJECTORIES = False
PLOT_SEABORN = False
# Long term: None for PLOT_TRAJECTORIES_GENERATION will automatically plot the trajectories for the final generation
PLOT_TRAJECTORIES_GENERATION = 500
# None for PLOT_TRAJECTORIES_TEAM_ID will automatically plot all of the trajectories for a generation
# on one big plot with a subplot for each joint-trajectory
# "Eval" for PLOT_TRAJECTORIES_TEAM_ID will plot the trajectories for the final evaluation team
PLOT_TRAJECTORIES_TEAM_ID = "Eval"

COMPUTERNAME = "playground"
TRIALNAME = getLatestTrialName(computername=COMPUTERNAME)

TRIALNAME =  "trial_4"

# Load in the trial data
save_data = loadTrial(TRIALNAME, COMPUTERNAME)
scores_list = save_data["scores_list"]
unfiltered_scores_list = save_data["unfiltered_scores_list"]
unfiltered_agent_scores_list = save_data["unfiltered_agent_scores_list"]
average_fitness_list_unfiltered = save_data["average_fitness_list_unfiltered"]
average_agent_fitness_lists_unfiltered = save_data["average_agent_fitness_lists_unfiltered"]
final_population = save_data["final_population"]
finished_iterations = save_data["finished_iterations"]
best_team_data = save_data["best_team_data"]
teams_in_evaluations = save_data["teams_in_evaluations"]
populations_through_generations = save_data["populations_through_generations"]
final_evaluation_teams = save_data["final_evaluation_teams"]

# Load in the config for that trial
config_filename = "config_" + TRIALNAME.split('_')[1] + ".yaml"
config = loadConfig(computername=COMPUTERNAME,config_name=config_filename)
config["CCEA"]["config"]["BoidsEnv"]["render_mode"] = "human"
leader_colors = config["CCEA"]["config"]["BoidsEnv"]["config"]["Renderer"]["leader_colors"]
leader_colors = tuple(np.array(leader_colors)/255)

# Functions that help with plotting
HEX_MAP = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
def map_digit_to_hex(digit: int)->str:
    # print(digit)
    return HEX_MAP[digit]

def rgb_to_hex(rgb_values: List[int]) -> str:
    # First each element (R G and B) needs to be converted to a hex string
    # Then just concatenate them together for a hex value
    hex_code = "#"
    for val in rgb_values:
        # print(val)
        first_digit = int(float(val)/16)
        second_digit = int( (float(val)%16. / 16.)*16. )
        # print(first_digit, second_digit)
        first_hex = map_digit_to_hex(first_digit)
        second_hex = map_digit_to_hex(second_digit)
        hex_code += first_hex 
        hex_code += second_hex
    return hex_code

def scaleRGB(rgb_values: List[int]):
    scaled_values = []
    for value in rgb_values:
        scaled_values.append(float(value)/255.)
    return tuple(scaled_values)

if PLOT_BEST_SCORES:
    plt.plot(scores_list, color="green", linestyle="--")
    plt.plot(unfiltered_scores_list, color="green")
    for ind, unfiltered_agent_scores in enumerate(unfiltered_agent_scores_list):
        plt.plot(unfiltered_agent_scores, color=tuple(leader_colors[ind%len(leader_colors)]))
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.ylim([0.0,1.01])
    plt.title("Best Performance")
    legend = ["Best Team Filtered", "Best Team Unfiltered"]
    for i in range(len(unfiltered_agent_scores_list)):
        legend.append("Agent "+str(i+1))
    plt.legend(legend)
    plt.show()

if PLOT_AVERAGE_SCORES:
    plt.plot(average_fitness_list_unfiltered, color="green")
    for ind, unfiltered_agent_scores in enumerate(average_agent_fitness_lists_unfiltered):
        plt.plot(unfiltered_agent_scores, color=tuple(leader_colors[ind%len(leader_colors)]))
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.ylim([0.0,1.2])
    plt.title("Average Performance")
    legend = ["Average Team Performance"]
    for i in range(len(average_agent_fitness_lists_unfiltered)):
        legend.append("Agent "+str(i+1))
    plt.legend(legend)
    plt.show()

if PLOT_FINAL_EVALUATION_TEAMS:
    # Get the fitnesses of the best teams
    final_evaluation_teams_fitnesses = [team_data.fitness for team_data in final_evaluation_teams]
    # Get the agent-specific scores for agents on each team
    num_leaders = config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"]
    agent_fitnesses = [[] for _ in range(num_leaders)]
    for team_data in final_evaluation_teams:
        for genome_id, genome_data in enumerate(team_data.team):
            agent_fitnesses[genome_id].append(genome_data.fitness)

    plt.plot(final_evaluation_teams_fitnesses , color="green")
    for ind, agent_specific_scores in enumerate(agent_fitnesses):
        plt.plot(agent_specific_scores, color=tuple(leader_colors[ind%len(leader_colors)]))
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.ylim([0.0,1.01])
    plt.title("Final Evaluation Teams")
    legend = ["Team Performance"]
    for i in range(len(agent_fitnesses)):
        legend.append("Agent "+str(i+1))
    plt.legend(legend)
    plt.show()

if PLAY_ENV:
    shutdown = False
    env = BoidsEnv(**config["CCEA"]["config"]["BoidsEnv"])
    dt = env.boids_colony.dt
    networks = [createNNfromWeights(genome_data.genome) for genome_data in best_team_data.team]
    print(best_team_data.fitness, best_team_data.all_evaluation_seeds)

    for eval_seed in best_team_data.all_evaluation_seeds:
        observations = env.reset(eval_seed)
        for step in range(env.max_steps):
            start_time = time()
            actions = {agent: computeAction(network, observations[agent], env) for agent, network in zip(env.possible_agents, networks)}
            observations, rewards, dones, infos = env.step(actions)
            env.render()
            # input()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Shutdown command recieved. Shutting down.")
                    shutdown = True
            if shutdown: exit()
            loop_time = time() - start_time
            if loop_time < dt:
                sleep(dt - loop_time)
            else:
                print("Loop " + str(step) + " took longer than refresh rate")
        # print("Team Fitness: ", env.fitness_calculator.getTeamFitness(), " | Agent Fitnesses: ", env.fitness_calculator.calculateDifferenceEvaluations())

def plotJointTrajectorySubplot(ax: Axes, generation: Optional[int], team_id: int | str, individual_plot: bool = True):
    if generation is None:
        generation = config["num_generations"]
    if team_id == "Eval":
        # Get the joint trajectory of the evaluation team
        joint_trajectory = np.array(final_evaluation_teams[generation].joint_trajectory).tolist()
    else:
        # First get the joint trajectory for this particular generation
        # Each element is a snapshot of all agent positions at a particular point in time
        # teams_in_evaluations is a global variable thanks to how python does things
        joint_trajectory = np.array(teams_in_evaluations[generation][team_id].joint_trajectory).tolist()
    
    # I need the joint trajectory as a list of trajectories where each trajectory is a list of (x,y) tuples for a particular agent
    num_trajectories = len(joint_trajectory[0])
    list_of_trajectories = [[] for _ in range(num_trajectories)]
    for positions in joint_trajectory:
        for position, trajectory in zip(positions, list_of_trajectories):
            trajectory.append(tuple(position))

    # Now I need to set up variables for the plot

    # Get map dimensions for figuring the x and y limits of the graph
    map_dimensions = config["CCEA"]["config"]["BoidsEnv"]["config"]["map_dimensions"]
    map_dim_x = map_dimensions["x"]
    map_dim_y = map_dimensions["y"]

    # Use map dimensions to figure out correctly proportioned graph size
    # Keep x dimension the same and adjust the y dimension accordingly
    # fig_x = 5.
    # fig_y = fig_x * float(map_dim_y)/float(map_dim_x)

    # Get the number of leaders and followers
    num_leaders = config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"]
    num_followers = config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_followers"]

    # Set up the leader colors for coloring the trajectories
    leader_colors_rgb_raw = config["CCEA"]["config"]["BoidsEnv"]["config"]["Renderer"]["leader_colors"]
    leader_colors_scaled = [scaleRGB(color) for color in leader_colors_rgb_raw]
    leader_colors = []
    for leader_ind in range(num_leaders):
        leader_colors.append(leader_colors_scaled[leader_ind%num_leaders])

    # Set up follower color
    follower_color = scaleRGB([0, 120, 250])
    # Set up colors of agents for all trajectories
    agent_colors = leader_colors + [follower_color]*num_followers

    # I reverse it so that the leader trajectories are plotted on top
    for trajectory, agent_color in reversed(list(zip(list_of_trajectories, agent_colors))):
        xs, ys = zip(*trajectory)
        ax.plot(xs, ys, color=agent_color)

    # Set up poi colors
    # For now just use the observed color because I don't yet save 
    # if a poi has been observed or not, so to determine that we would have to do a rollout or 
    # call some code to compute that
    poi_observed_color = scaleRGB([0, 150, 0])
    # poi_unobserved_color = scaleRGB([255, 0, 0])

    # Get the POI positions for the configuration
    # Later on this should plot the poi observation radius also
    poi_positions = config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"]
    num_pois = len(poi_positions)
    poi_colors = [poi_observed_color]*num_pois
    for poi_position, poi_color in zip(poi_positions, poi_colors):
        ax.plot(poi_position[0], poi_position[1], marker=".", color=poi_color)

    # Add axes labels and a title
    if individual_plot:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Joint Trajectory')

    # Limits according to map dimensions from config
    ax.set_xlim([0, map_dim_x])
    ax.set_ylim([0, map_dim_y])

    # Custom legend
    leader_handles = [
        Line2D([0], [0], color=leader_color, lw=1) for leader_color in leader_colors
    ]
    leader_labels = ["Leader "+str(i+1) for i in range(num_leaders)]

    follower_handle = Line2D([0], [0], color=follower_color, lw=1)
    follower_label = "Follower"

    poi_handle = Line2D([0], [0], color=poi_observed_color, lw=0, marker=".")
    poi_label = "POI"

    handles = leader_handles + [follower_handle] + [poi_handle]
    labels = leader_labels + [follower_label] + [poi_label]

    if individual_plot:
        ax.legend(handles, labels)

    # Add a grid and make it look pretty
    ax.grid()
    ax.tick_params(
        axis='both',
        which='both',
        top = False,
        bottom = False,
        left = False,
        right = False
    )
    # Give plot a gray background like ggplot.
    ax.set_facecolor('#EBEBEB')
    # Remove border around plot.
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # Style the grid.
    ax.grid(which='major', color='white', linewidth=1.2)
    ax.grid(which='minor', color='white', linewidth=0.6)

    # Set up the ticks for the grid
    xticks = np.linspace(0, int(map_dim_x - (map_dim_x%10)), int(map_dim_x/10.)+1)
    yticks = np.linspace(0, int(map_dim_y - (map_dim_y%10)), int(map_dim_y/10.)+1)

    # Remove labels for individual ticks if option is specified
    if not individual_plot:
        ax.set_xticks(ticks=xticks, labels=[])
        ax.set_yticks(ticks=yticks, labels=[])

    # Show the minor ticks and grid.
    # ax.minorticks_on()
    # Now hide the minor ticks (but leave the gridlines).
    # ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_aspect('equal')

if PLOT_TRAJECTORIES:
    # If no team id is specified, then plot all of the joint trajectories for this generation
    if PLOT_TRAJECTORIES_TEAM_ID is None:
        # Figure out the grid size to place all of the plots in
        # Reference: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
        num_teams = config["CCEA"]["sub_population_size"]
        grid_len = int(np.ceil(np.sqrt(num_teams)))
        fig, axs = plt.subplots(nrows=grid_len, ncols=grid_len, figsize=(15,15), tight_layout=True)
        for team_id, ax in zip(np.arange(num_teams), axs.flat):
            plotJointTrajectorySubplot(ax=ax, generation=PLOT_TRAJECTORIES_GENERATION, team_id=team_id, individual_plot=False)
            # Objects for a custom legend that just lets me display important metadata
            class AnyObject:
                pass
            class AnyObjectHandler:
                def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                    x0, y0 = handlebox.xdescent, handlebox.ydescent
                    width, height = handlebox.width, handlebox.height
                    patch = mpatches.Rectangle([x0, y0], 0, 0, facecolor='red',
                                            edgecolor='black', hatch='xx', lw=0,
                                            transform=handlebox.get_transform())
                    handlebox.add_artist(patch)
                    return patch
                
            # Extract the fitness for this particular team
            team_fitness = teams_in_evaluations[PLOT_TRAJECTORIES_GENERATION][team_id].fitness

            # Format that fitness into a nice str
            fitness_str = f"{team_fitness:.3f}"

            # Custom legend that acts as a label for what team this plot is from
            # fake_handle = Line2D([0], [0], color='white', lw=0)
            fake_handle = AnyObject()
            fake_label = str(team_id) + " | " + fitness_str
            ax.legend([fake_handle], [fake_label], loc='upper right', handler_map={AnyObject: AnyObjectHandler()}, handlelength=-1)
            fig.suptitle("Generation "+str(PLOT_TRAJECTORIES_GENERATION))
        
        for ax in axs.flat[team_id+1:]:
            ax.tick_params(
                axis='both',
                which='both',
                top = False,
                bottom = False,
                left = False,
                right = False
            )
            ax.set_xticks(ticks=[], labels=[])
            ax.set_yticks(ticks=[], labels=[])
            # Remove border around plot.
            [ax.spines[side].set_visible(False) for side in ax.spines]

        fig.subplots_adjust(wspace=0, hspace=0)

        plt.show()
        pass
    # If team id is specified, then just plot that one joint trajectory
    else:
        # Get map dimensions for figuring the x and y limits of the graph
        map_dimensions = config["CCEA"]["config"]["BoidsEnv"]["config"]["map_dimensions"]
        map_dim_x = map_dimensions["x"]
        map_dim_y = map_dimensions["y"]

        # Use map dimensions to figure out correctly proportioned graph size
        # Keep x dimension the same and adjust the y dimension accordingly
        fig_x = 5.
        fig_y = fig_x * float(map_dim_y)/float(map_dim_x)

        # Set up the plot
        fig = plt.figure(figsize=(fig_x, fig_y))
        ax = fig.add_subplot(1,1,1)
        plotJointTrajectorySubplot(ax=ax, generation=PLOT_TRAJECTORIES_GENERATION, team_id=PLOT_TRAJECTORIES_TEAM_ID)
        plt.show()

if PLOT_SEABORN:
    # Set up variables for seaborn plotting
    labels = ['t','x', 'y', 'name', 'leader', 'poi', 'observed', 'Label']
    data = []

    # Get the trajectories for a particular generation
    # Each element is a snapshot of all agent positions at a particular point in time
    joint_trajectory = teams_in_evaluations[PLOT_TRAJECTORIES_GENERATION][PLOT_TRAJECTORIES_TEAM_ID].joint_trajectory

    # We also want to include the initial step
    number_steps = len(joint_trajectory)

    # Get the number of leaders
    num_leaders = config["CCEA"]["config"]["BoidsEnv"]["config"]["StateBounds"]["num_leaders"]

    # Get the POI positions for the configuration
    poi_positions = config["CCEA"]["config"]["BoidsEnv"]["config"]["POISpawner"]["positions"]

    for t in range(number_steps):
        # Aggregate and append all of the agent info
        # This takes the snapshot of the joint trajectory at time t which is an array of positions for
        # the different agents
        for agent_id, position in enumerate(joint_trajectory[t]):
            # Figure out if its a leader or not
            if agent_id < num_leaders:
                label = "Leader "+str(agent_id)
                isLeader = True
                agent_name = label
            else:
                label = "Follower"
                isLeader = False
                agent_name = "Follower "+str(agent_id-num_leaders)
            data.append([t, position[0], position[1], agent_name, isLeader, False, False, label])

        # Aggregate and append all of the poi
        for poi_ind, poi_position in enumerate(poi_positions):
            poi_name = "POI "+str(poi_ind)
            label = "POI"
            # Not going to worry about whether they were observed or not for now
            data.append([t, poi_position[0], poi_position[1], poi_name, False, True, True, label])

    traj_df = DataFrame(data=data, columns=labels)

    # pp.pprint(joint_trajectory)

    # Get map dimensions for figuring the x and y limits of the graph
    map_dimensions = config["CCEA"]["config"]["BoidsEnv"]["config"]["map_dimensions"]
    map_dim_x = map_dimensions["x"]
    map_dim_y = map_dimensions["y"]

    # Use map dimensions to figure out correctly proportioned graph size
    # Keep x dimension the same and adjust the y dimension accordingly
    graph_x = 10.
    graph_y = graph_x * float(map_dim_y)/float(map_dim_x)

    # Create the color palette so the plot uses what is in the configuration file
    leader_colors_rgb = config["CCEA"]["config"]["BoidsEnv"]["config"]["Renderer"]["leader_colors"]
    # print(leader_colors_rgb)
    leader_colors_hex = [rgb_to_hex(leader_color_rgb) for leader_color_rgb in leader_colors_rgb]
    leader_colors_pallette = []
    for leader_ind in range(num_leaders):
        # Need to do this weird circular remainder step in order to make sure we have the correct number of colors
        # in the color pallete for the number of leaders we have
        # If we have more leaders than colors, we want to reuse colors. Same idea as when we render with pygame.
        leader_colors_pallette.append(leader_colors_hex[leader_ind%num_leaders])

    print(leader_colors_pallette)

    # sns.set_palette(sns.color_palette(leader_colors_pallette))
    sns.set_palette("Set1")

    # Create the seaborn plot object
    seaborn_plot = so.Plot(data=traj_df, x="x", y="y", color="Label", marker="name")
    # seaborn_plot = so.Plot(data=traj_df, x="x", y="y", marker="name")

    # Create the path object for all the paths of agents and pois (pois stay still)
    seaborn_path = so.Path(marker="o", pointsize=1.5, linewidth=1)
    # Add the paths to the plot
    seaborn_plot = seaborn_plot.add(mark=seaborn_path)

    # Add limits to the plot
    seaborn_plot = seaborn_plot.limit(y=(0, map_dim_y),x=(0, map_dim_x))

    # Add layout for the sizing
    seaborn_plot.layout(size=(graph_x, graph_y))

    # Show the final plot
    seaborn_plot.show()

    # and whether that POI was observed or not
    # Long term: also get the observation radius of the POI for plotting circles around them



    # data.append([t, state])