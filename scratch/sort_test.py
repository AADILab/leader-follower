# sorted_pop = [genome for _, _, genome in sorted(zip(scores, list(range(len(population))), population))]
# sorted_pop.reverse()
import random

def resetPopulation():
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]

population = resetPopulation()
scores =     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
num_iterations = 7

print("0: ", population)

def sortPopulation(scores, population):
    sorted_pop = [genome for _, _, genome in sorted(zip(scores, list(range(len(population))), population))]
    sorted_pop.reverse()
    return sorted_pop

def mutatePopulation(population):
    children = [random.choice(population[:5])[-1]+child for child in population[5:]]
    parents = [parent[-1]+parent for parent in population[:5]]
    # parents = population[:5]
    # children = [random.choice(parents) for _ in range(10)]
    return parents+children

for _ in range(num_iterations):
    population = mutatePopulation(population)
    population = sortPopulation(scores, population)

print("1: ", population)

population = resetPopulation()

def sortPopulationBetter(scores, population):
    sorted_pop = [genome for _, genome in sorted(zip(scores, population), reverse=True)]
    sorted_scores = [score for score in sorted(scores, reverse=True)]

    start_inds = [0]
    end_inds = []
    last_score = None

    for ind, score in enumerate(sorted_scores):
        if last_score is None:
            last_score = score
        elif score != last_score:
            last_score = score
            start_inds.append(ind)
            end_inds.append(ind)

    end_inds.append(15)

    for start_ind, end_ind in zip(start_inds, end_inds):
        shuffled_genomes = sorted_pop[start_ind:end_ind]
        random.shuffle(shuffled_genomes)
        sorted_pop[start_ind:end_ind] = shuffled_genomes

    # print(sorted_pop)

    return sorted_pop

for _ in range(num_iterations):
    population = mutatePopulation(population)
    population = sortPopulationBetter(scores, population)

print("2: ", population)

# def mutatePopulationBetter(population):
#     return [random.choice(population)[-1]+genome for genome in population]

# population = resetPopulation()

# for _ in range(num_iterations):
#     population = mutatePopulationBetter(population)
#     population = sortPopulation(scores, population)

# print("3: ", population)

# population = resetPopulation()

# for _ in range(num_iterations):
#     population = mutatePopulationBetter(population)

# print("4: ", population)