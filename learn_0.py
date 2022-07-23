from learner_lib import Learner

learner = Learner(population_size=15, num_parents=5, sigma_mutation=0.1, FPS=30, num_steps=30*30)
scores = learner.train(num_generations=10)
print(scores)
