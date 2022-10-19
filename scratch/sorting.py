import random

population = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
scores =     [0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]

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

print(start_inds)
print(end_inds)

for start_ind, end_ind in zip(start_inds, end_inds):
    shuffled_genomes = sorted_pop[start_ind:end_ind]
    random.shuffle(shuffled_genomes)
    sorted_pop[start_ind:end_ind] = shuffled_genomes

print(sorted_pop)
print(sorted_scores)
# print(unique_scores)
