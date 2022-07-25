from learner_lib import Learner
from time import time

start = time()

learner = Learner(population_size=15, num_parents=5, sigma_mutation=0.1)
scores = learner.train(num_generations=1)
# print(scores)

print(time() - start)
