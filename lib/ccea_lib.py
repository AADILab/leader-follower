import queue
import traceback
from copy import copy, deepcopy
from multiprocessing import Event, Process, Queue
from typing import List, Dict, Optional, Any
import time
import sys
import os

import numpy as np
from tqdm import tqdm
import pandas as pd

from lib.boids_env import BoidsEnv
from lib.network_lib import NN

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
    def __init__(self, team: List[GenomeData], id: int, fitness: Optional[float] = None,
                 evaluation_seed: int = 0) -> None:
        super().__init__()
        self.team = team
        self.id = id
        self.fitness = fitness
        self.difference_evaluations = []
        self.evaluation_seed = evaluation_seed
        self.all_evaluation_seeds = []
        # Joint trajectory includes leaders and followers
        self.joint_trajectory = None


def computeAction(net, observation, env):
    out = net.forward(observation)
    # Map [-1,+1] to [-pi,+pi]
    heading = out[0] * np.pi
    # Map [-1,+1] to [0, max_velocity]
    velocity = (out[1] + 1.0) / 2 * env.state_bounds.max_velocity
    return np.array([heading, velocity])


class EvaluationWorker:
    def __init__(self, in_queue, out_queue, stop_event: Event, id: int, team_size: int,
                 num_evaluations: int, env_kwargs: Dict = None, nn_kwargs: Dict = None):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.id = id
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
                    start_evaluate_team_time = time.time()
                    team_data.fitness, team_data.difference_evaluations, team_data.joint_trajectory = self.evaluateTeam(team_data, False)
                    stop_evaluate_team_time = time.time()
                    interval_evaluate_team_time = stop_evaluate_team_time - start_evaluate_team_time

                except AttributeError as e:
                    print(f"AttributeError on EvaluationWorker {self.id}")
                    print(traceback.format_exc())
                    team_data.fitness = 0

                self.out_queue.put(team_data)

        except KeyboardInterrupt:
            print(f"Interrupt on EvaluationWorker {self.id}!")
            self.stop_event.set()
        except Exception as e:
            print(
                f"Error on EvaluationWorker {self.id}! Exiting program. Error: {e}\nFull Traceback:\n{traceback.format_exc()}")
            self.stop_event.set()
        finally:
            print(f"Shutting down EvaluationWorker {self.id}")

    def setupTeamPolicies(self, team_data: TeamData):
        for genome_data, net in zip(team_data.team, self.team_policies):
            net.setWeights(genome_data.genome)

    def evaluateTeam(self, team_data: TeamData, draw: bool = False) -> tuple[Any, list[Any], list[np.ndarray]]:
        """Load team into boids environment and calculate a fitness score."""
        self.setupTeamPolicies(team_data)

        team_data.all_evaluation_seeds = [team_data.evaluation_seed + n for n in range(self.num_evaluations)]

        fitnesses = np.zeros((self.num_evaluations, 1 + self.env.num_agents))

        traj = np.zeros((self.env.max_steps + 1, 2))

        for eval_count, evaluation_seed in enumerate(team_data.all_evaluation_seeds):
            # Run network on boids environment
            observations = self.env.reset(seed=evaluation_seed)
            traj[self.env.num_steps] = self.env.boids_colony.state.positions[0]
            done = False
            while not done:
                if draw:
                    self.env.render()
                # Collect the action for each agent on the team
                actions = {agent_name: computeAction(net, observations[agent_name], self.env) for agent_name, net in
                           zip(self.env.possible_agents, self.team_policies)}
                # Step forward the environment
                observations, rewards, dones, _ = self.env.step(actions)
                # Save done
                done = True in dones.values()
                traj[self.env.num_steps] = self.env.boids_colony.state.positions[0]
            self.env.close()

            fitnesses[eval_count] = np.array([rewards["team"]] + [rewards[agent] for agent in self.env.agents])

        team_fitness = np.average(fitnesses[:, 0])
        agent_fitnesses = [np.average(fitnesses[:, num_agent + 1]) for num_agent in range(self.env.num_agents)]
        return team_fitness, agent_fitnesses, self.env.position_history


class CCEA:
    def __init__(self,
                sub_population_size: int,
                mutation_rate: float, mutation_probability: float,
                nn_hidden: int,
                num_workers: int,
                num_evaluations: int,
                # This is for when initial state is random. Evaluating several times ensures we dont
                # just take policies that happen to get lucky with an easy start.
                config: Dict,
                trial_path: str,
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
        self.genome_uid = 0

        # Variables for saving data throughout training
        self.teams_in_evaluation = []
        self.populations_through_generations = []
        self.final_evaluation_teams = []
        self.trial_path = trial_path

        # Setup nn variables
        self.nn_inputs = config["BoidsEnv"]["config"]["ObservationManager"]["num_poi_bins"] + \
                         config["BoidsEnv"]["config"]["ObservationManager"]["num_swarm_bins"]
        self.nn_hidden = nn_hidden
        self.nn_outputs = 2
        if init_population is None:
            self.population = self.randomPopulation()
        else:
            self.population = init_population
        self.fitnesses = self.initFitnesses()

        # Setup object for deterministic random seed generation
        # self.seed_generator = SeedGenerator()

        # Process event - set flag to True to turn off workers
        self.stop_event = Event()

        self.work_queue = Queue(1000)
        self.fitness_queue = Queue(1000)
        init_workers = self.initEvaluationWorkers()
        self.workers = self.setupEvaluationWorkers(init_workers)
        self.startEvaluationWorkers()

    def generateUid(self):
        _id = self.genome_uid
        self.genome_uid += 1
        return _id

    def randomGenome(self):
        # Create a NN with random weights and get the weights as the genome
        weights = NN(num_inputs=self.nn_inputs, num_hidden=self.nn_hidden, num_outputs=self.nn_outputs).getWeights()

        # print('--')
        # for weight in weights:
        #     print(weight, end=',')
        # print('\n--')
        return weights

    def randomSubPopulation(self):
        return [
            GenomeData(self.randomGenome(), gid=id, uid=self.generateUid())
            for id in range(self.sub_population_size)
        ]

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

    def mutateGenome(self, genome: Genome, seed: int = None) -> Genome:
        """Mutate weights of genome with zero-mean gaussian noise."""
        # Make sure seed is random for mutations
        # if seed is None:
        #     seed = generateSeed()
        # np.random.seed(seed)

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
    def copySubPop(sub_population: List[GenomeData]) -> List[GenomeData]:
        c = []
        for genome_data in sub_population:
            c.append(genome_data)
        return c

    def randomTeams(self, evaluation_seed: Optional[int] = None):
        # Form random teams from sub populations
        random_teams = [
            TeamData(team=[], id=id, evaluation_seed=evaluation_seed) for id in range(self.sub_population_size)
        ]
        # Shuffle subpopulations for random ordering of policies in each subpopulation
        shuffled_sub_populations = [copy(sub_pop) for sub_pop in self.population]

        for sub_pop in shuffled_sub_populations:
            np.random.shuffle(sub_pop)
        for sub_population in shuffled_sub_populations:
            for team_ind, genome_data in enumerate(sub_population):
                random_teams[team_ind].team.append(genome_data)

        return random_teams

    def evaluatePopulation(self):
        # Generate random seed for evaluation.
        # This ensures each genome from a specific population is evaluated on the same task.
        # Otherwise, one team might seem better than others just because it solved an easy task.
        # evaluation_seed = self.generateSeed()
        evaluation_seed = np.random.randint(0, 100)

        # Form random teams from population
        random_teams = self.randomTeams(evaluation_seed=evaluation_seed)

        # Send teams to Evaluation Workers for evaluation
        for team_data in random_teams:
            self.work_queue.put(team_data)

        # Keep track of which teams have been recieved after evaluation
        receieved = [False for _ in random_teams]
        timeout = 10  # seconds
        self.teams: List[TeamData] = [None for _ in random_teams]

        while not all(receieved) and not self.stop_event.is_set():
            try:
                # Grab latest evaluated team from workers
                team_data = self.fitness_queue.get(timeout=timeout)
                # Store the team
                self.teams[team_data.id] = team_data
                # Update received
                receieved[team_data.id] = True
            except queue.Empty:
                pass

        # Go back and assign fitnesses for genomes on teams. This is necessary for keeping metadata consistent.
        # The teams evaluated by the workers contain copies of the genomes, not the originals, meaning we have to
        # manually update the fitnesses of genomes on teams.

        for evaluated_team_data in self.teams:
            for agent_id, genome_data in enumerate(evaluated_team_data.team):
                genome_data.fitness = evaluated_team_data.difference_evaluations[agent_id]
                # Save the trajectory that was taken by this particular policy in its team evaluation
                # This makes an assumption about how the joint trajectory is organized
                # Leader trajectories are all on the left-hand side of the array, followers on the right
                genome_data.trajectory = np.array(evaluated_team_data.joint_trajectory)[:, agent_id]

        # This array is helpful for debugging and ensuring that difference evaluations
        # are assigned correctly to each policy
        covered = [[0 for _ in sub_pop] for sub_pop in self.population]

        # Assign fitnesses from teams to agent policies on that team
        for evaluated_team_data in self.teams:
            # Each team is ordered by agent id. And each genome has an id corresponding
            # to its position in the sub population
            for agent_id, genome_data in enumerate(evaluated_team_data.team):
                self.population[agent_id][genome_data.id].fitness = evaluated_team_data.difference_evaluations[agent_id]
                covered[agent_id][genome_data.id] += 1

        # Run one more team evaluation using the highest scoring policy from each sub population on the same team
        # Get the highest scoring policy from each subpopulation
        highest_scoring_genome_datas = []
        for sub_population in self.population:
            # GenomeData has a builtin for sorting by fitness
            # This should give us the GenomeData with the highest fitness
            best_genome_data_ind = sub_population.index(max(sub_population))
            best_genome_data = sub_population[best_genome_data_ind]
            highest_scoring_genome_datas.append(best_genome_data)

        # print(best_genome_data)
        # import sys; sys.exit()
        # Use these genomes to create a team
        evaluation_team_data_in = TeamData(team=highest_scoring_genome_datas, id=self.sub_population_size)
        # Put this team data into the queue for evaluation
        self.work_queue.put(evaluation_team_data_in)
        # Seperate team_data_in and team_data_out. I think this might be pass by value rather than pointer
        # from looking at how I did this earlier
        evaluation_recieved = False
        while not evaluation_recieved and not self.stop_event.is_set():
            try:
                # Grab the evaluated team
                evaluation_team_data_out = self.fitness_queue.get(timeout=timeout)
                # Save that we received it to break this loop
                evaluation_recieved = True
            except queue.Empty:
                pass
        # Save the evaluated team data
        self.final_evaluation_teams.append(deepcopy(evaluation_team_data_out))

        # Save all the team data during evaluation
        self.teams_in_evaluation.append(deepcopy(self.teams))
        self.populations_through_generations.append(deepcopy(self.population))

        # Save the team with the highest fitness. Both a filtered one and the current best
        self.teams.sort(reverse=True)
        self.current_best_team_data = deepcopy(self.teams[0])
        if self.best_team_data is None or self.teams[0].fitness > self.best_team_data.fitness:
            self.best_team_data = deepcopy(self.teams[0])
            # print("Team Fitness: ", self.best_team_data.fitness, " | Agent Fitnesses: ",
            #       [genome_data.fitness for genome_data in self.best_team_data.team])

    def downSelectPopulation(self):
        """Take a population which has already been evaluated and create a new population
        for the next generation with n-elites binary tournament"""
        new_population = [[] for _ in range(self.num_agents)]

        # Run binary tournament for each sub population
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
                # Grab 2 policies from within this sub-population at random
                genome_a, genome_b = np.random.choice(self.population[n_agent], 2, replace=False)
                # Get the genome with the highest fitness
                genome_winner = [genome_a, genome_b][np.argmax([genome_a, genome_b])]
                # Mutate that genome and add it to the new population
                mutated_genome = GenomeData(
                    genome=self.mutateGenome(genome_winner.genome),
                    gid=len(new_population[n_agent]),
                    uid=self.generateUid()
                )
                new_population[n_agent].append(mutated_genome)

        # Replace population with newly selected population for next generation
        self.population = new_population

    def savePopulation(self, generation_dir: str):
        """ Save all policies from all subpopulations.
        Each leader has all its policies saved.
        Each policy is saved as an npz file
        """

        # Make directory for population
        population_dir = os.path.join(generation_dir, "population")
        os.makedirs(population_dir)

        for leader_ind, subpopulation in enumerate(self.population):
            leader_name = "leader_"+str(leader_ind)
            leader_subpop_dir = os.path.join(population_dir, leader_name)
            os.makedirs(leader_subpop_dir)
            for genome_data in subpopulation:
                # Save the weights (list of np arrays) as npz file
                policy_filename = "policy_" + str(genome_data.id) + ".npz"
                npz_file_dir = os.path.join(leader_subpop_dir, policy_filename)
                # Turn into a dictionary for saving more easily as npz
                weights_dict = {"layer_"+str(ind) : weight_matrix for ind, weight_matrix in enumerate(genome_data.genome)}
                np.savez_compressed(npz_file_dir, *weights_dict)

    def saveTrainingTeams(self, generation_dir: str):
        """Save the scores for the agents and teams. Save the id of which policy each agent used"""
        training_teams_dir = os.path.join(generation_dir, "training_teams")
        os.makedirs(training_teams_dir)
        for team_data in self.teams:
            # Get the directory for this team
            team_dir = os.path.join(training_teams_dir, "team_"+str(team_data.id))
            # Make that directory
            os.makedirs(team_dir)

            # Get out the fitness of the team and individual agents
            team_fitness = team_data.fitness
            agent_fitnesses = team_data.difference_evaluations
            # Get out the id of which policy was used for each agent
            policy_ids = [genome_data.id for genome_data in team_data.team]
            # Put this all into a data frame so we can save it as a csv
            fitness_data = {}
            for leader_id in range(len(team_data.team)):
                fitness_data[str(leader_id)] = [policy_ids[leader_id], agent_fitnesses[leader_id]]
            fitness_data["G"] = [0, team_fitness]

            # Custom defined index
            index = ["policy_id", "fitness"]

            # Put everything in a dataframe
            df = pd.DataFrame(data=fitness_data, index=index)
            df.columns.name = "leader_id"
            
            # Get name of csv file
            csv_dir = os.path.join(team_dir, "fitnesses.csv")

            # Save that as a csv
            df.to_csv(path_or_buf=csv_dir)

            # # Get the joint trajectory that this team took
            # joint_trajectory = np.array(team_data.joint_trajectory)
            
            # position_headers = []
            # for leader_id in range(self.num_agents):
            

            # for positions 

            

    def saveEvaluationTeam(self):
        pass

    def saveGeneration(self):
        """ Save all of the data associated with this generation """
        # Make directory for this generation
        generation_name = "generation_"+str(self.iterations)
        generation_dir = os.path.join(self.trial_path, generation_name)
        os.makedirs(generation_dir)

        # Save the population
        self.savePopulation(generation_dir)

        # Save the training teams
        self.saveTrainingTeams(generation_dir)

        # Save the evaluation team
        self.saveEvaluationTeam()

        return None

    def step(self):
        # Select genomes for next generation
        self.downSelectPopulation()

        # Evaluate the population
        # todo check if seed is set before here
        self.evaluatePopulation()

        # Increase iterations counter
        self.iterations += 1

        # Save the generation
        # This needs to be called after we increase the iterations counter
        # Otherwise on the first call to step(), saveGeneration() will try to overwrite 
        # what we saved for iteration 0
        self.saveGeneration()

    def getFinalMetrics(self):
        return self.best_fitness_list, self.best_fitness_list_unfiltered, self.best_agent_fitness_lists_unfiltered, \
               self.average_fitness_list_unfiltered, self.average_agent_fitness_lists_unfiltered, \
               self.population, self.iterations, self.best_team_data, \
               self.teams_in_evaluation, self.populations_through_generations, self.final_evaluation_teams

    def saveFitnesses(self):
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

    def train(self, num_generations: int):
        """Train the learner for a set number of generations. Track performance data."""
        # Evaluate the initial random policies
        # todo check if seed is set before here
        self.evaluatePopulation()
        # evaluatePopulation() is saving all of the policy/team/trajectory data

        self.saveGeneration()

        # Save fitnesses for initial random policies as generation 0
        self.saveFitnesses()
        
        for _ in tqdm(range(num_generations)):
            self.step()
            # step() calls evaluatePopulation() so this saves all of the policy/team/trajectory data at each timestep

            # Track fitness over time
            self.saveFitnesses()
        return None
