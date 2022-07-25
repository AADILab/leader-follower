from learner_lib import Learner
from time import time

start = time()

learner = Learner(population_size=15, num_parents=5, sigma_mutation=0.1, env_kwargs={"num_leaders": 10, "num_followers": 90, "FPS": 60, "num_steps": 60*60, "render_mode": 'none'})
scores = learner.train(num_generations=1)
learner.cleanup()
# print(scores)

print(time() - start)
