import queue
import traceback
from copy import copy, deepcopy
from multiprocessing import Event, Process, Queue
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm

from leader_follower.boids_env import BoidsEnv
from leader_follower.network_lib import NN

# Genome encodes weights of a network as list of numpy arrays
Genome = List[np.array]


class SortByFitness:

    def __init__(self):
        self.fitness = None
        return

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, SortByFitness):
            return self.fitness == __o.fitness

    def __ne__(self, __o: object) -> bool:
        if isinstance(__o, SortByFitness):
            return self.fitness != __o.fitness

    def __lt__(self, __o: object) -> bool:
        if isinstance(__o, SortByFitness):
            return self.fitness < __o.fitness

    def __le__(self, __o: object) -> bool:
        if isinstance(__o, SortByFitness):
            return self.fitness <= __o.fitness

    def __gt__(self, __o: object) -> bool:
        if isinstance(__o, SortByFitness):
            return self.fitness > __o.fitness

    def __ge__(self, __o: object) -> bool:
        if isinstance(__o, SortByFitness):
            return self.fitness >= __o.fitness


class GenomeData(SortByFitness):
    def __init__(self, genome: Genome, gid: int, fitness: Optional[float] = None, uid: Optional[int] = None) -> None:
        super().__init__()
        self.genome = genome
        self.id = gid
        self.fitness = fitness
        self.uid = uid


class TeamData(SortByFitness):
    def __init__(self, team: List[GenomeData], team_id: int, fitness: Optional[float] = None,
                 evaluation_seed: Optional[int] = None) -> None:
        super().__init__()
        self.team = team
        self.id = team_id
        self.fitness = fitness
        self.difference_evaluations = []
        self.evaluation_seed = evaluation_seed
        self.all_evaluation_seeds = []


def compute_action(net, observation, env):
    out = net.forward(observation)
    # Map [-1,+1] to [-pi,+pi]
    heading = out[0] * np.pi
    # Map [-1,+1] to [0, max_velocity]
    velocity = (out[1] + 1.0) / 2 * env.state_bounds.max_velocity
    return np.array([heading, velocity])


class EvaluationWorker:
    def __init__(self, in_queue, out_queue, stop_event: Event, worker_id: int, team_size: int,
                 use_difference_rewards: bool, num_evaluations: int, env_kwargs: Dict = None, nn_kwargs: Dict = None):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.id = worker_id
        self.use_difference_rewards = use_difference_rewards
        self.num_evaluations = num_evaluations
        self.env = BoidsEnv(**env_kwargs)
        self.team_policies = [NN(**nn_kwargs) for _ in range(team_size)]

    def __call__(self):
        try:
            while not self.stop_event.is_set():
                try:
                    team_data = self.in_queue.get(timeout=0.01)
                    # print("Worker ",self.id," just received team_data ", id(team_data))
                except queue.Empty:
                    continue

                try:
                    team_data.fitness, team_data.difference_evaluations = self.evaluate_team(team_data, False)
                except AttributeError as e:
                    print(f"AttributeError on EvaluationWorker {self.id}")
                    print(f'{e}')
                    print(traceback.format_exc())
                    team_data.fitness = 0

                self.out_queue.put(team_data)

        except KeyboardInterrupt:
            print(f"Interrupt on EvaluationWorker {self.id}!")
            self.stop_event.set()
        except Exception as e:
            print(f"Error on EvaluationWorker {self.id}! Exiting program. Error: {e}\n"
                  f"Full Traceback:\n"
                  f"{traceback.format_exc()}"
                  )
            self.stop_event.set()
        finally:
            print(f"Shutting down EvaluationWorker {self.id}")

    def setup_team_policies(self, team_data: TeamData):
        for genome_data, net in zip(team_data.team, self.team_policies):
            net.set_weights(genome_data.genome)

    def evaluate_team(self, team_data: TeamData, draw: bool = False):
        """Load team into boids environment and calculate a fitness score."""
        self.setup_team_policies(team_data)

        team_data.all_evaluation_seeds = [team_data.evaluation_seed + n for n in range(self.num_evaluations)]

        fitnesses = np.zeros((self.num_evaluations, 1 + self.env.num_agents))

        traj = np.zeros((self.env.max_steps + 1, 2))

        for eval_count, evaluation_seed in enumerate(team_data.all_evaluation_seeds):
            # Run network on boids environment
            observations = self.env.reset(seed=evaluation_seed)
            traj[self.env.num_steps] = self.env.boids_colony.state.positions[0]
            done = False
            rewards = {}
            while not done:
                if draw:
                    self.env.render()
                # Collect the action for each agent on the team
                actions = {agent_name: compute_action(net, observations[agent_name], self.env) for agent_name, net in
                           zip(self.env.possible_agents, self.team_policies)}
                # Step forward the environment
                observations, rewards, dones, _ = self.env.step(actions)
                # Save done
                done = True in dones.values()
                traj[self.env.num_steps] = self.env.boids_colony.state.positions[0]
            self.env.close()

            if self.use_difference_rewards:
                fitnesses[eval_count] = np.array([rewards["team"]] + [rewards[agent] for agent in self.env.agents])
            else:
                fitnesses[eval_count] = rewards["team"] * np.ones(1 + len(self.env.agents))

        team_fitness = np.average(fitnesses[:, 0])
        agent_fitnesses = [np.average(fitnesses[:, num_agent + 1]) for num_agent in range(self.env.num_agents)]
        return team_fitness, agent_fitnesses


class CCEA:
    def __init__(self,
                 sub_population_size: int,
                 mutation_rate: float, mutation_probability: float,
                 nn_hidden: int,
                 use_difference_evaluations: bool,
                 num_workers: int,
                 num_evaluations: int,
                 # This is for when initial state is random. Evaluating several times ensures we don't
                 # just take policies that happen to get lucky with an easy start.
                 config: Dict,
                 init_population=None,
                 ) -> None:
        # Set variables
        self.num_agents = config["BoidsEnv"]["config"]["StateBounds"]["num_leaders"]
        self.sub_population_size = sub_population_size
        self.sigma_mutation = mutation_rate
        self.mutation_probability = mutation_probability
        self.iterations = 0
        self.num_workers = num_workers
        self.num_evaluations = num_evaluations
        self.config = config
        self.best_fitness_list = []
        self.best_fitness_list_unfiltered = []
        self.best_agent_fitness_lists_unfiltered = [[] for _ in range(self.num_agents)]
        self.average_fitness_list_unfiltered = []
        self.average_agent_fitness_lists_unfiltered = [[] for _ in range(self.num_agents)]
        self.best_team_data = None
        self.current_best_team_data = None
        self.use_difference_rewards = use_difference_evaluations
        self.genome_uid = 0
        self.teams: List[Optional[TeamData]] = []

        # Setup nn variables
        self.nn_inputs = config["BoidsEnv"]["config"]["ObservationManager"]["num_poi_bins"] + \
            config["BoidsEnv"]["config"]["ObservationManager"]["num_swarm_bins"]
        self.nn_hidden = nn_hidden
        self.nn_outputs = 2
        if init_population is None:
            self.population = self.random_population()
        else:
            self.population = init_population
        self.fitnesses = self.init_fitnesses()

        # Setup object for deterministic random seed generation
        # self.seed_generator = SeedGenerator()

        # Process event - set flag to True to turn off workers
        self.stop_event = Event()

        self.work_queue = Queue(1000)
        self.fitness_queue = Queue(1000)
        init_workers = self.init_eval_workers()
        self.workers = self.setup_eval_workers(init_workers)
        self.start_eval_workers()
        return

    def generate_uid(self):
        _id = self.genome_uid
        self.genome_uid += 1
        return _id

    def random_genome(self):
        # Create a NN with random weights and get the weights as the genome
        weights = NN(num_inputs=self.nn_inputs, num_hidden=self.nn_hidden, num_outputs=self.nn_outputs).get_weights()
        return weights

    def random_subpopulation(self):
        return [
            GenomeData(self.random_genome(), gid=gid, uid=self.generate_uid())
            for gid in range(self.sub_population_size)
        ]

    def random_population(self):
        return [self.random_subpopulation() for _ in range(self.num_agents)]

    def init_subpop_fitnesses(self):
        return [None for _ in range(self.sub_population_size)]

    def init_fitnesses(self):
        return [self.init_subpop_fitnesses() for _ in range(self.num_agents)]

    def init_eval_workers(self):
        return [
            EvaluationWorker(
                in_queue=self.work_queue,
                out_queue=self.fitness_queue,
                stop_event=self.stop_event,
                worker_id=worker_id,
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

    @staticmethod
    def setup_eval_workers(init_workers):
        return [
            Process(
                target=worker,
                args=(),
            )
            for worker in init_workers
        ]

    def start_eval_workers(self):
        for w in self.workers:
            w.start()
            return

    def __del__(self):
        self.cleanup()
        return

    def cleanup(self):
        # End work processes on deletion
        try:
            self.stop_event.set()
            for w in self.workers:
                w.join()
        except Exception as e:
            print(f'ccea cleanup {e}')

    def mutate_genome(self, genome: Genome, seed: int = None) -> Genome:
        """Mutate weights of genome with zero-mean gaussian noise."""
        # Make sure seed is random for mutations
        np.random.seed(seed)
        new_genome = []
        for layer in genome:
            new_layer = deepcopy(layer)
            rand = np.reshape(
                np.random.uniform(low=0.0, high=1.0, size=new_layer.size),
                new_layer.shape
            )
            weight_multipliers = np.random.normal(0.0, self.sigma_mutation,
                                                  size=new_layer[rand < self.mutation_probability].size)
            new_layer[rand < self.mutation_probability] += weight_multipliers * new_layer[
                rand < self.mutation_probability]
            new_genome.append(new_layer)
        return new_genome

    @staticmethod
    def copy_subpop(sub_population: List[GenomeData]) -> List[GenomeData]:
        c = []
        for genome_data in sub_population:
            c.append(genome_data)
        return c

    def random_teams(self, evaluation_seed: Optional[int] = None):
        # Form random teams from subpopulations
        random_teams = [
            TeamData(team=[], team_id=pop_id, evaluation_seed=evaluation_seed)
            for pop_id in range(self.sub_population_size)
        ]
        # Shuffle subpopulations for random ordering of policies in each subpopulation
        shuffled_sub_populations = [copy(sub_pop) for sub_pop in self.population]

        for sub_pop in shuffled_sub_populations:
            np.random.shuffle(sub_pop)
        for sub_population in shuffled_sub_populations:
            for team_ind, genome_data in enumerate(sub_population):
                random_teams[team_ind].team.append(genome_data)

        return random_teams

    def eval_population(self):
        # Generate random seed for evaluation.
        # This ensures each genome from a specific population is evaluated on the same task.
        # Otherwise, one team might seem better than others just because it solved an easy task.
        # evaluation_seed = self.generateSeed()
        evaluation_seed = np.random.randint(0, 100)

        # Form random teams from population
        random_teams = self.random_teams(evaluation_seed=evaluation_seed)

        # Send teams to Evaluation Workers for evaluation
        for team_data in random_teams:
            self.work_queue.put(team_data)

        # Keep track of which teams have been received after evaluation
        received = [False for _ in random_teams]
        timeout = 10  # seconds
        self.teams: List[TeamData] = [None for _ in random_teams]

        while not all(received) and not self.stop_event.is_set():
            try:
                # Grab latest evaluated team from workers
                team_data = self.fitness_queue.get(timeout=timeout)
                # Store the team
                self.teams[team_data.id] = team_data
                # Update received
                received[team_data.id] = True
            except queue.Empty:
                pass

        # Go back and assign fitnesses for genomes on teams. This is necessary for keeping metadata consistent.
        # The teams evaluated by the workers contain copies of the genomes, not the originals, meaning we have to
        # manually update the fitnesses of genomes on teams.

        for evaluated_team_data in self.teams:
            for agent_id, genome_data in enumerate(evaluated_team_data.team):
                genome_data.fitness = evaluated_team_data.difference_evaluations[agent_id]

        # This array is helpful for debugging and ensuring that difference evaluations
        # are assigned correctly to each policy
        covered = [[0 for _ in sub_pop] for sub_pop in self.population]

        # Assign fitnesses from teams to agent policies on that team
        for evaluated_team_data in self.teams:
            # Each team is ordered by agent id. And each genome has an id corresponding
            # to its position in the subpopulation
            for agent_id, genome_data in enumerate(evaluated_team_data.team):
                if self.use_difference_rewards:
                    self.population[agent_id][genome_data.id].fitness = evaluated_team_data.difference_evaluations[
                        agent_id]
                else:
                    self.population[agent_id][genome_data.id].fitness = evaluated_team_data.fitness
                covered[agent_id][genome_data.id] += 1

        # Save the team with the highest fitness. Both a filtered one and the current best
        self.teams.sort(reverse=True)
        self.current_best_team_data = deepcopy(self.teams[0])
        if self.best_team_data is None or self.teams[0].fitness > self.best_team_data.fitness:
            self.best_team_data = deepcopy(self.teams[0])
            # print("Team Fitness: ", self.best_team_data.fitness, " | Agent Fitnesses: ",
            #       [genome_data.fitness for genome_data in self.best_team_data.team])
        return self.best_team_data.fitness, [genome_data.fitness for genome_data in self.best_team_data.team]

    def downselect(self):
        """Take a population which has already been evaluated and create a new population
        for the next generation with n-elites binary tournament"""
        new_population = [[] for _ in range(self.num_agents)]

        # Run binary tournament for each subpopulation
        for n_agent in range(self.num_agents):
            # First store n unmutated highest scoring policies
            n = 1
            self.population[n_agent].sort(reverse=True)
            new_population[n_agent] += deepcopy(self.population[n_agent][:n])
            # Make sure ids are consistent
            for _id, genome_data in enumerate(new_population[n_agent]):
                genome_data.id = _id
            # Generate new policies until we have the correct number of policies
            while len(new_population[n_agent]) < self.sub_population_size:
                # Grab 2 policies from within this subpopulation at random
                genome_a, genome_b = np.random.choice(self.population[n_agent], 2, replace=False)
                # Get the genome with the highest fitness
                genome_winner = [genome_a, genome_b][np.argmax([genome_a, genome_b])]
                # Mutate that genome and add it to the new population
                mutated_genome = GenomeData(
                    genome=self.mutate_genome(genome_winner.genome),
                    gid=len(new_population[n_agent]),
                    uid=self.generate_uid()
                )
                new_population[n_agent].append(mutated_genome)

        # Replace population with newly selected population for next generation
        self.population = new_population

    def step(self):
        # Select genomes for next generation
        self.downselect()

        # Evaluate the population
        fitnesses = self.eval_population()

        # Increase iterations counter
        self.iterations += 1
        return fitnesses

    def final_metrics(self):
        return self.best_fitness_list, self.best_fitness_list_unfiltered, self.best_agent_fitness_lists_unfiltered, \
               self.average_fitness_list_unfiltered, self.average_agent_fitness_lists_unfiltered, \
               self.population, self.iterations, self.best_team_data

    def save_fitnesses(self):
        # Save bests
        self.best_fitness_list.append(self.best_team_data.fitness)
        self.best_fitness_list_unfiltered.append(self.current_best_team_data.fitness)
        # average team performance
        self.average_fitness_list_unfiltered.append(np.average([team_data.fitness for team_data in self.teams]))
        # Save best fitness of each individual agent in this generation
        for agent_id in range(self.num_agents):
            fitnesses = [genome_data.fitness for genome_data in self.population[agent_id]]
            # Best fitness
            self.best_agent_fitness_lists_unfiltered[agent_id].append(max(fitnesses))
            # Average fitness
            self.average_agent_fitness_lists_unfiltered[agent_id].append(np.average(fitnesses))
        return

    def train(self, num_generations: int):
        """Train the learner for a set number of generations. Track performance data."""
        # Evaluate the initial random policies
        self.eval_population()

        # Save fitnesses for initial random policies as generation 0
        self.save_fitnesses()
        with tqdm(total=num_generations) as pbar:
            for idx in range(num_generations):
                self.step()
                # Track fitness over time
                fitnesses = [genome_data.fitness for genome_data in self.best_team_data.team]
                fitness_str = f'Team fitness: {self.best_team_data.fitness} | Agent Fitnesses: {fitnesses}'
                pbar.set_postfix_str(fitness_str)
                pbar.update(1)
        return None
