import queue
from typing import List, Dict, Optional, Callable, Tuple, Union
import random
import traceback
from multiprocessing import Event, Process, Queue
from time import time
from copy import copy, deepcopy

from tqdm import tqdm
import numpy as np

from lib.network_lib import NN
from lib.boids_env import BoidsEnv

# Genome encodes weights of a network as list of numpy arrays
Genome = List[np.array]

class SortByFitness():
    def __eq__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness == __o.fitness

    def __ne__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness != __o.fitness

    def __lt__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness < __o.fitness

    def __le__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness <= __o.fitness

    def __gt__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness > __o.fitness

    def __ge__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness >= __o.fitness


class GenomeData(SortByFitness):
    def __init__(self, genome: Genome, id: int, fitness: Optional[float]=None) -> None:
        self.genome = genome
        self.id = id
        self.fitness = fitness

class TeamData(SortByFitness):
    def __init__(self, team: List[GenomeData], id: int, fitness: Optional[float]=None, evaluation_seed: Optional[int]=None) -> None:
        self.team = team
        self.id = id
        self.fitness = fitness
        self.difference_evaluations = []
        self.evaluation_seed = evaluation_seed
        self.all_evaluation_seeds = []

def generateSeed():
    return int((time() % 1) * 1000000)

def computeAction(net, observation, env):
    out = net.forward(observation)
    # Map [-1,+1] to [-pi,+pi]
    heading = out[0] * np.pi
    # Map [-1,+1] to [0, max_velocity]
    velocity = (out[1]+1.0)/2*env.state_bounds.max_velocity
    return np.array([heading, velocity])

class EvaluationWorker():
    def __init__(self, in_queue, out_queue, stop_event: Event, id: int, team_size: int, use_difference_rewards: bool, num_evaluations: int, env_kwargs: Dict = {}, nn_kwargs: Dict ={}):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.id = id
        self.use_difference_rewards = use_difference_rewards
        self.num_evaluations = num_evaluations
        self.env = BoidsEnv(**env_kwargs)
        self.team_policies = [NN(**nn_kwargs) for _ in range(team_size)]

    def __call__(self):
        try:
            while not self.stop_event.is_set():
                try:
                    team_data = self.in_queue.get(timeout=0.01)
                except queue.Empty:
                    continue

                try:
                    team_data.fitness, team_data.difference_evaluations = self.evaluateTeam(team_data, False)
                except AttributeError as e:
                    print(f"AttributeError on EvaluationWorker {self.id}")
                    print(traceback.format_exc())
                    team_data.fitness = 0

                self.out_queue.put(team_data)

        except KeyboardInterrupt:
            print(f"Interrupt on EvaluationWorker {self.id}!")
            self.stop_event.set()
        except Exception as e:
            print(f"Error on EvaluationWorker {self.id}! Exiting program. Error: {e}\nFull Traceback:\n{traceback.format_exc()}")
            self.stop_event.set()
        finally:
            print(f"Shutting down EvaluationWorker {self.id}")

    def setupTeamPolicies(self, team_data: TeamData):
        for genome_data, net in zip(team_data.team, self.team_policies):
            net.setWeights(genome_data.genome)

    def evaluateTeam(self, team_data: TeamData, draw: bool = False) -> float:
        """Load team into boids environment and calculate a fitness score."""
        # Load networks with weights from genomes on team
        self.setupTeamPolicies(team_data)

        team_data.all_evaluation_seeds = [team_data.evaluation_seed+n for n in range(self.num_evaluations)]

        fitnesses = np.zeros((self.num_evaluations, 1+self.env.num_agents))

        for eval_count, evaluation_seed in enumerate(team_data.all_evaluation_seeds):
            # Run network on boids environment
            observations = self.env.reset(seed=evaluation_seed)
            done = False
            while not done:
                if draw:
                    self.env.render()
                # Collect the action for each agent on the team
                actions = {agent_name: computeAction(net, observations[agent_name], self.env) for agent_name, net in zip(self.env.possible_agents, self.team_policies)}
                # Step forward the environment
                observations, rewards, dones, _  = self.env.step(actions)
                # Save done
                done = True in dones.values()
            self.env.close()
            fitnesses[eval_count] = np.array([rewards["team"]]+[rewards[agent] for agent in self.env.agents])
        team_fitness = np.average(fitnesses[:,0])
        agent_fitnesses = [np.average(fitnesses[:,num_agent+1]) for num_agent in range(self.env.num_agents)]
        # return rewards["team"], [rewards[self.env.possible_agents[agent_id]] for agent_id in range(self.env.num_agents)]
        return team_fitness, agent_fitnesses

class CCEA():
    def __init__(self,
        sub_population_size: int, num_parents: int,
        mutation_rate: float, mutation_probability: float,
        nn_hidden: int,
        num_workers: int = 4,
        num_evaluations: int = 1, # This is for when initial state is random. Evaluating several times ensures we dont just take policies that happen to get lucky with an easy start.
        init_population = None,
        use_difference_evaluations: bool = True,
        config: Dict = {}
        ) -> None:
        # Set variables
        self.num_agents = config["BoidsEnv"]["config"]["StateBounds"]["num_leaders"]
        self.sub_population_size = sub_population_size
        self.num_parents = num_parents
        self.num_children = sub_population_size - num_parents
        self.sigma_mutation = mutation_rate
        self.mutation_probability = mutation_probability
        self.iterations = 0
        self.num_workers = num_workers
        self.num_evaluations = num_evaluations
        self.config = config
        self.best_fitness_list = []
        self.best_team_data = None
        self.use_difference_rewards = use_difference_evaluations

        # Setup nn variables
        self.nn_inputs = config["BoidsEnv"]["config"]["ObservationManager"]["num_poi_bins"] + config["BoidsEnv"]["config"]["ObservationManager"]["num_swarm_bins"]
        self.nn_hidden = nn_hidden
        self.nn_outputs = 2
        if init_population is None:
            self.population = self.randomPopulation()
        else:
            self.population = init_population
        self.fitnesses = self.initFitnesses()

        # Process event - set flag to True to turn off workers
        self.stop_event = Event()

        self.work_queue = Queue(1000)
        self.fitness_queue = Queue(1000)
        init_workers = self.initEvaluationWorkers()
        self.workers = self.setupEvaluationWorkers(init_workers)
        self.startEvaluationWorkers()

    def randomGenome(self):
        # Create a NN with random weights and get the weights as the genome
        return NN(num_inputs=self.nn_inputs, num_hidden=self.nn_hidden, num_outputs=self.nn_outputs).getWeights()

    def randomSubPopulation(self):
        return [GenomeData(self.randomGenome(), id=id) for id in range(self.sub_population_size)]

    def randomPopulation(self):
        return [self.randomSubPopulation() for _ in range(self.num_agents)]

    def initSubPopFitnesess(self):
        return [None for _ in range(self.sub_population_size)]

    def initFitnesses(self):
        return [self.initSubPopFitnesess() for _ in range(self.num_agents)]

    def initEvaluationWorkers(self):
        return [
        EvaluationWorker(
            in_queue=self.work_queue,
            out_queue=self.fitness_queue,
            stop_event=self.stop_event,
            id=worker_id,
            use_difference_rewards=self.use_difference_rewards,
            num_evaluations=self.num_evaluations,
            env_kwargs=self.config["BoidsEnv"],
            team_size=self.num_agents,
            nn_kwargs={
                "num_inputs": self.nn_inputs,
                "num_hidden": self.nn_hidden,
                "num_outputs": self.nn_outputs
            }
        )
        for worker_id in range(self.num_workers)
    ]

    def setupEvaluationWorkers(self, init_workers):
        return [
            Process(
                target=worker,
                args=(),
            )
            for worker in init_workers
        ]

    def startEvaluationWorkers(self):
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

    def mutateGenome(self, genome: Genome) -> Genome:
        # """Mutate weights of genome with zero-mean gaussian noise."""
        new_genome = []
        for layer in genome:
            new_layer = layer.copy()
            rand = np.reshape(np.random.uniform(low=0.0,high=1.0,size=new_layer.size), new_layer.shape)
            # print(rand.shape, new_layer.shape, layer.shape)
            new_layer[rand < self.mutation_probability] += np.random.normal(0.0, self.sigma_mutation, size=new_layer[rand<self.mutation_probability].size)*new_layer[rand < self.mutation_probability]
            new_genome.append(new_layer)
            # new_genome.append(layer + np.random.normal(0.0, self.sigma_mutation, size=(layer.shape)))
        return new_genome

    def randomTeams(self, evaluation_seed: Optional[int] = None):
        # Form random teams from sub populations
        random_teams = [
            TeamData(team=[], id=id, evaluation_seed=evaluation_seed) for id in range(self.sub_population_size)
        ]

        # Shuffle subpopulations for random ordering of policies in each subpopulation
        shuffled_sub_populations = [copy(sub_pop) for sub_pop in self.population]
        for sub_pop in shuffled_sub_populations:
            random.shuffle(sub_pop)
        for sub_population in shuffled_sub_populations:
            for team_ind, genome_data in enumerate(sub_population):
                random_teams[team_ind].team.append(genome_data)

        return random_teams

    def evaluatePopulation(self):
        # Generate random seed for evaluation.
        # This ensures each genome from a specific population is evaluated on the same task.
        # Otherwise, one team might seem better than others just because it solved an easy task.
        evaluation_seed = generateSeed()

        # Form random teams from population
        random_teams = self.randomTeams(evaluation_seed = evaluation_seed)

        # Send teams to Evaluation Workers for evaluation
        for team_data in random_teams:
            self.work_queue.put(team_data)

        # Keep track of which teams have been recieved after evaluation
        receieved = [False for _ in random_teams]
        timeout = 10 # seconds
        evaluated_teams = []

        while not all(receieved) and not self.stop_event.is_set():
            try:
                # Grab latest evaluated team from workers
                team_data = self.fitness_queue.get(timeout=timeout)
                # Store the team
                evaluated_teams.append(team_data)
                # Update received
                receieved[team_data.id] = True
            except queue.Empty:
                pass

        covered = [[0 for _ in sub_pop] for sub_pop in self.population]

        # Assign fitnesses from teams to agent policies on that team
        for evaluated_team_data in evaluated_teams:
            # Each team is ordered by agent id. And each genome has an id corresponding to its position in the sub population
            for agent_id, genome_data in enumerate(evaluated_team_data.team):
                if self.use_difference_rewards:
                    self.population[agent_id][genome_data.id].fitness = evaluated_team_data.difference_evaluations[agent_id]
                else:
                    self.population[agent_id][genome_data.id].fitness = evaluated_team_data.fitness
                covered[agent_id][genome_data.id] += 1

        # Save the team with the highest fitness
        evaluated_teams.sort(reverse=True)
        if self.best_team_data is None or evaluated_teams[0].fitness > self.best_team_data.fitness:
            self.best_team_data = deepcopy(evaluated_teams[0])
            print(self.best_team_data.fitness, self.best_team_data.evaluation_seed)

    def mutatePopulation(self):
        # Mutate policies
        for sub_pop in self.population:
            # Sort each sub population so highest fitness policies are at the front
            sub_pop.sort(reverse=True)
            # Set new ids for parents since their position in the sub population changed
            for new_id, parent_data in enumerate(sub_pop[:self.num_parents]):
                parent_data.id = new_id
            # Mutate the highest scoring policies and replace low scoring policies
            sub_pop[self.num_parents:] = [GenomeData(genome=self.mutateGenome(random.choice(sub_pop[self.num_parents:]).genome), id=self.num_parents+child_num) for child_num in range(self.num_children)]

    def step(self):
        # Mutate the population
        self.mutatePopulation()
        # Evaluate the population
        self.evaluatePopulation()
        # Increase iterations counter
        self.iterations += 1

    def getFinalMetrics(self):
        return self.best_fitness_list, self.population, self.iterations, self.best_team_data

    def train(self, num_generations: int):
        """Train the learner for a set number of generations. Track performance data."""
        # Evaluate the initial random policies
        self.evaluatePopulation()
        # Track fitness over time
        self.best_fitness_list.append(self.best_team_data.fitness)
        for _ in tqdm(range(num_generations)):
            self.step()
            self.best_fitness_list.append(self.best_team_data.fitness)
        return None
