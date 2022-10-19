from time import time
import numpy as np

import sys; sys.path.append("/home/egonzalez/leaders")
from lib.map_utils import Map

# Map should be faster at retrieving nearby boids in the map. Let's compare against numpy's built-in boolean functions
# First setup the map and the boid positions
total_boids = 5
observation_radius = 2
map_size = np.array([100,100])
positions = np.hstack((
                np.random.uniform(map_size[0], size=(total_boids,1)),
                np.random.uniform(map_size[1], size=(total_boids,1))
            ))

# This sets up the map and uses it to check which boids are close to which boids
start_map = time()
boids_map = Map(map_size, observation_radius, positions)
for b in range(total_boids):
    boids_map.get_observable_agent_inds(positions[b], positions)
stop_map = time()

# This just checks using numpy methods
start_np = time()
for position in positions:
    distances = np.linalg.norm(positions - position , axis=1)
    positions[distances<=observation_radius]
stop_np = time()

print("Map time: ", stop_map-start_map, " | Numpy time: ", stop_np-start_np)
# Numpy seems to be an order of magnitude faster than my custom map. Rip map.
