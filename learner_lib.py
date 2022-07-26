from ast import Call
import queue
from typing import List, Dict, Optional, Callable
import random
import traceback
from sys import exit

# import torch
# import torch.nn as nn
from tqdm import tqdm
import numpy as np
from multiprocessing import Event, Process, Queue, Manager
from time import time
from network_lib import NN

from env_lib import BoidsEnv

# Genome encodes weights of a network as list of tensors
# Genome = List[torch.TensorType]
Genome = List[np.array]

def generateSeed():
    return int((time() % 1) * 1000000)

class Worker():
    def __init__(self, in_queue: Queue, out_queue: Queue, stop_event: Event, id: int, env_kwargs: Dict = {}, nn_kwargs: Dict ={}):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.id = id
        self.env = BoidsEnv(**env_kwargs)
        self.net = NN(**nn_kwargs)
        self.genome_id = None

    def __call__(self):
        try:
            while not self.stop_event.is_set():
                try:
                    input = self.in_queue.get(timeout=0.01)
                except queue.Empty:
                    continue

                self.genome_id = input["id"]

                genome = input["genome"]
                seed = input["seed"]

                try:
                    # fitness = self.evaluateGenome(genome, seed, False)
                    fitness = self.evaluateGenome(genome, seed, False)

                except AttributeError as e:
                    print("AttributeError!")
                    print(e)
                    fitness = np.inf

                output = {"id": input["id"], "fitness": fitness}
                self.out_queue.put(output)

        except KeyboardInterrupt:
            print(f"Interrupt on Worker {self.id}, Genome {self.genome_id} !")
        except Exception as e:
            print(f"Error on Worker {self.id}, Genome {self.genome_id}! Exiting program. Error: {e}\nFull Traceback:\n{traceback.format_exc()}")
            self.stop_event.set()
        finally:
            print(f"Shutting down Worker {self.id}")

    def evaluateGenome(self, genome: Genome, seed: int = 0, draw: bool = False) -> float:
        """Load genome into boids environment and calculate a fitness score."""
        # Load network with weights from genome
        self.net.setWeights(genome)

        # Run network on boids environment
        observations = self.env.reset()
        done = False
        cumulative_reward = 0
        while not done:
            if draw:
                self.env.render()
            # Collect actions for all agents with each agent using the same genome to guide its action
            actions = {agent_name: self.net.forward(np.array(observations[agent_name])) for agent_name in self.env.possible_agents}
            # Step forward the environment
            observations, rewards, dones, _  = self.env.step(actions)
            # Save done
            done = True in dones.values()
            # Add the team reward to the cumulative reward
            # Need [0] index because rewards are an array of rewards. One for each objective.
            cumulative_reward += rewards["team"][0]
        self.env.close()

        return cumulative_reward

class Learner():
    def __init__(self, population_size: int, num_parents: int, sigma_mutation: float, num_workers: int = 10, env_kwargs: Dict = {"num_leaders": 10, "num_followers": 90, "FPS": 60, "num_steps": 60*60, "render_mode": 'none'}) -> None:
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

        # Initialize environment
        self.env_kwargs = env_kwargs

        # Process event - set flag to True to turn off workers
        self.stop_event = Event()

        self.work_queue = Queue(1000)
        self.fitness_queue = Queue(1000)
        init_workers = [
            Worker(
                in_queue=self.work_queue,
                out_queue=self.fitness_queue,
                stop_event=self.stop_event,
                id=worker_id,
                env_kwargs=env_kwargs,
                nn_kwargs={"num_inputs": self.input_size, "num_hidden": self.hidden_size, "num_outputs": self.out_size}
            )
            for worker_id in range(num_workers)
        ]
        self.workers = [
            Process(
                target=worker,
                args=(),
            )
            for worker in init_workers
        ]
        for w in self.workers:
            w.start()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        # End work processes on deletion
        try:
            self.stop_event.set()
            for w in self.workers:
                w.join()
        except:
            pass

    def randomGenome(self):
        # Create a NN with random weights and get the weights as the genome
        return NN(num_inputs=self.input_size, num_hidden=self.hidden_size, num_outputs=self.out_size).getWeights()

    def mutateGenome(self, genome: Genome) -> Genome:
        """Mutate weights of genome with zero-mean gaussian noise."""
        new_genome = []
        for layer in genome:
            new_genome.append(layer + np.random.normal(0.0, self.sigma_mutation, size=(layer.shape)))
        return new_genome

    def mutatePopulation(self, scores) -> List[Genome]:
        """Generate a new population based on the fitness scores of the genomes in the population."""
        # Sort population so that lowest scoring genomes are at the front of the list
        sorted_population = [genome for _, _, genome in sorted(zip(scores, list(range(len(self.population))), self.population))]

        # Trying to minimize distance of swarm to objective, so lower scores are better
        # Keep parents with lowest scores
        parents = sorted_population[:self.num_parents]
        # Randomly select parents and mutate them to get the rest of the population
        children = [self.mutateGenome(random.choice(parents)) for _ in range(self.num_children)]
        # Return a new population with the best-fit parents and mutated children
        mutated_population = parents + children
        return mutated_population

    def evaluatePopulation(self, population = None):
        if population is None:
            population = self.population

        # Queue genomes to be evaluated by workers
        seed = generateSeed()
        for genome_id, genome in enumerate(population):
            self.work_queue.put({"id": genome_id, "genome": genome, "seed": seed})

        # Keep track of which genomes' fitnesses have been received
        received = [False for _ in population]
        timeout = 10 # seconds
        fitnesses = [np.inf for _ in population]

        while not all (received) and not self.stop_event.is_set():
            try:
                # Grab results from workers
                rx = self.fitness_queue.get(timeout=timeout)

                id = rx["id"]
                fit = rx["fitness"]

                # Match received fitness to the corresponding genome
                received[id] = True
                fitnesses[id] = fit
            except queue.Empty:
                print("Timeout reached on waiting for a response!")
                print(
                    f"Currently received {sum(received)} out of {len(received)} responses!"
                )
                print(
                    f"Work: {self.work_queue.qsize()}, Fitnesses: {self.fitness_queue.qsize()}"
                )
        if self.stop_event.is_set():
            print("Stop event detected. Exiting main program.")
            exit()
        return fitnesses

    def step(self) -> float:
        """Step forward the learner by a generation and update the population."""
        # Evaluate all the genomes in the population
        # scores = [self.evaluateGenome(genome) for genome in self.population]
        fitnesses = self.evaluatePopulation()
        # Mutate the population according to the fitness scores
        self.population = self.mutatePopulation(fitnesses)
        return min(fitnesses)

    def train(self, num_generations: int):
        """Train the learner for a set number of generations. Save performance data."""
        score_list = []
        for _ in tqdm(range(num_generations)):
            min_score = self.step()
            score_list.append(min_score)
        return score_list
