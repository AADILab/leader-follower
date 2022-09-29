from typing import Dict

import pygame

from lib.boids_env import BoidsEnv
from lib.ccea_lib import CCEA, TeamData, computeAction
from lib.network_lib import NN, createNNfromWeights
from time import time, sleep

def watchConfig(config: Dict):
    config["CCEA"]["config"]["BoidsEnv"]["render_mode"] = "human"
    config["CCEA"]["num_workers"] = 0

    # Initialize a CCEA. This makes it easer to grab certain parameters we need
    # Then create a random network for each learner
    ccea = CCEA(**config["CCEA"])
    networks = [NN(num_inputs=ccea.nn_inputs, num_hidden=ccea.nn_hidden, num_outputs=ccea.nn_outputs) for _ in range(ccea.num_agents)]

    shutdown = False
    env = BoidsEnv(**config["CCEA"]["config"]["BoidsEnv"])
    dt = env.boids_colony.dt

    observations = env.reset()
    for step in range(env.max_steps):
        start_time = time()
        actions = {agent: computeAction(network, observations[agent], env) for agent, network in zip(env.agents, networks)}
        observations, rewards, dones, infos = env.step(actions)
        env.render()
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

def watchTeam(config: Dict, team_data: TeamData):
    shutdown = False
    env = BoidsEnv(**config["CCEA"]["config"]["BoidsEnv"])
    dt = env.boids_colony.dt
    networks = [createNNfromWeights(genome_data.genome) for genome_data in team_data.team]
    print(team_data.fitness, team_data.all_evaluation_seeds)

    for eval_seed in team_data.all_evaluation_seeds:
        observations = env.reset(eval_seed)
        for step in range(env.max_steps):
            start_time = time()
            actions = {agent: computeAction(network, observations[agent], env) for agent, network in zip(env.possible_agents, networks)}
            observations, rewards, dones, infos = env.step(actions)
            env.render()
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
        print("Team Fitness: ", env.fitness_calculator.getTeamFitness(), " | Agent Fitnesses: ", env.fitness_calculator.calculateDifferenceEvaluations())
    print(env.boids_colony.state.positions)