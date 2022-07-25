from typing import List, Dict
import random

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, Queue, Manager

from env_lib import BoidsEnv

# Genome encodes weights of a network as list of tensors
Genome = List[torch.TensorType]

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hid_size, out_size) -> None:
        super(FeedForwardNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, out_size),
            nn.Tanh()
        )
        self.model.requires_grad_(False)

    def run(self, x) -> torch.TensorType:
        return self.model(x)

    def getWeights(self) -> Genome:
        d = self.model.state_dict()
        return [d['0.weight'], d['2.weight']]

    def setWeights(self, weights: Genome) -> None:
        d = self.model.state_dict()
        d['0.weight'] = weights[0]
        d['2.weight'] = weights[1]
        self.model.load_state_dict(d)

class Learner():
    def __init__(self, population_size: int, num_parents: int, sigma_mutation: float, num_workers: int = 4, env_kwargs: Dict = {"num_leaders": 10, "num_followers": 90, "FPS": 60, "num_steps": 60*60, "render_mode": 'none'}) -> None:
        # Set variables
        self.population_size = population_size
        self.num_parents = num_parents
        self.num_children = population_size - num_parents
        self.sigma_mutation = sigma_mutation

        # Initialize population
        self.input_size = 4
        self.hidden_size = 6
        self.out_size = 2
        self.population = [self.randomGenome() for _ in range(self.population_size)]

        # self.work_queue = Queue(1000)
        # self.fitness_queue = Queue(1000)
        # self.workers = [
        #     Process(
        #         target=self.evalutationWorker,
        #         args=(self.work_queue, self.fitness_queue, i, env_kwargs),
        #     )
        #     for i in range(num_workers)
        # ]

        # Initialize environment
        self.env = BoidsEnv(**env_kwargs)

    def createNet(self) -> FeedForwardNet:
        return FeedForwardNet(self.input_size, self.hidden_size, self.out_size)

    def randomGenome(self) -> Genome:
        return [
            torch.normal(mean=0.0, std=1.0, size=(self.hidden_size, self.input_size)),
            torch.normal(mean=0.0, std=1.0, size=(self.out_size, self.hidden_size))
            ]

    # def evaluationWorker(in_queue, out_queue, id=0, env_kwargs={}):
    #     try:
    #         env = BoidsEnv(**env_kwargs)
    #         while True:
    #             input = in_queue.
    #     except KeyboardInterrupt:
    #         print(f"Interrupt! Exiting {id}")

    def evaluateGenome(self, genome: Genome, draw: bool = False) -> float:
        """Load genome into boids environment and calculate a fitness score."""
        # Load network with weights from genome
        net = self.createNet()
        net.setWeights(genome)

        # Run network on boids environment
        observations = self.env.reset()
        done = False
        cumulative_reward = 0
        while not done:
            if draw:
                self.env.render()
            # Collect actions for all agents with each agent using the same genome to guide its action
            actions = {agent_name: net.run(torch.tensor(np.array([observations[agent_name]]), dtype=torch.float)) for agent_name in self.env.possible_agents}
            # Step forward the environment
            observations, rewards, dones, _  = self.env.step(actions)
            # Save done
            done = True in dones.values()
            # Add the team reward to the cumulative reward
            # Need [0] index because rewards are an array of rewards. One for each objective.
            cumulative_reward += rewards["team"][0]
        self.env.close()
        return cumulative_reward

    def mutateGenome(self, genome: Genome) -> Genome:
        """Mutate weights of genome with zero-mean gaussian noise."""
        new_genome = []
        for layer in genome:
            new_genome.append(layer + torch.normal(mean=0.0, std=self.sigma_mutation, size=(layer.shape)))
        return new_genome

    def mutatePopulation(self, scores) -> List[Genome]:
        """Generate a new population based on the fitness scores of the genomes in the population."""
        # Sort population so that lowest scoring genomes are at the front of the list
        sorted_population = [genome for _, genome in sorted(zip(scores, self.population))]
        # Trying to minimize distance of swarm to objective, so lower scores are better
        # Keep parents with lowest scores
        parents = sorted_population[:self.num_parents]
        # Randomly select parents and mutate them to get the rest of the population
        children = [self.mutateGenome(random.choice(parents)) for _ in range(self.num_children)]
        # Return a new population with the best-fit parents and mutated children
        mutated_population = parents + children
        return mutated_population

    # def evaluatePopulation(self, population = None):
    #     if population is None:
    #         population = self.population



    def step(self) -> float:
        """Step forward the learner by a generation and update the population."""
        # Evaluate all the genomes in the population
        scores = [self.evaluateGenome(genome) for genome in self.population]
        # Mutate the population according to the fitness scores
        self.population = self.mutatePopulation(scores)
        return min(scores)

    def train(self, num_generations: int):
        """Train the learner for a set number of generations. Save performance data."""
        score_list = []
        for _ in tqdm(range(num_generations)):
            min_score = self.step()
            score_list.append(min_score)
        return score_list
