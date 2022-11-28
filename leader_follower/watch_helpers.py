from typing import Dict

import pygame

from leader_follower.boids_env import BoidsEnv
from leader_follower.ccea_lib import CCEA, TeamData, compute_action
from leader_follower.network_lib import NN, create_nn_from_weights
from time import time, sleep


def watch_config(config: Dict):
    config["CCEA"]["config"]["BoidsEnv"]["render_mode"] = "human"
    config["CCEA"]["num_workers"] = 0

    # Initialize a CCEA. This makes it easier to grab certain parameters we need
    # Then create a random network for each learner
    ccea = CCEA(**config["CCEA"])
    networks = [NN(num_inputs=ccea.nn_inputs, num_hidden=ccea.nn_hidden, num_outputs=ccea.nn_outputs) for _ in
                range(ccea.num_agents)]

    shutdown = False
    env = BoidsEnv(**config["CCEA"]["config"]["BoidsEnv"])
    dt = env.boids_colony.dt

    observations = env.reset()
    for step in range(env.max_steps):
        start_time = time()
        actions = {agent: compute_action(network, observations[agent], env) for agent, network in
                   zip(env.agents, networks)}
        observations, rewards, dones, infos = env.step(actions)
        print(list(observations["leader_0"][0:8]))
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Shutdown command received. Shutting down.")
                shutdown = True
        if shutdown:
            exit()
        loop_time = time() - start_time
        if loop_time < dt:
            sleep(dt - loop_time)
        else:
            print("Loop " + str(step) + " took longer than refresh rate")
    return


def watch_team(config: Dict, team_data: TeamData):
    shutdown = False
    env = BoidsEnv(**config["CCEA"]["config"]["BoidsEnv"])
    dt = env.boids_colony.dt
    networks = [create_nn_from_weights(genome_data.genome) for genome_data in team_data.team]
    print(team_data.fitness, team_data.all_evaluation_seeds)

    for eval_seed in team_data.all_evaluation_seeds:
        observations = env.reset(eval_seed)
        for step in range(env.max_steps):
            start_time = time()
            actions = {agent: compute_action(network, observations[agent], env) for agent, network in
                       zip(env.possible_agents, networks)}
            observations, rewards, dones, infos = env.step(actions)
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Shutdown command received. Shutting down.")
                    shutdown = True
            if shutdown:
                exit()
            loop_time = time() - start_time
            if loop_time < dt:
                sleep(dt - loop_time)
            else:
                print("Loop " + str(step) + " took longer than refresh rate")
        print("Team Fitness: ", env.fitness_calculator.global_discrete(), " | Agent Fitnesses: ",
              env.fitness_calculator.difference_evaluations())
    print(env.boids_colony.state.positions)
    return
