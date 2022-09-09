import numpy as np
from time import sleep

from lib.boids_colony import BoidsColony
from lib.env_renderer import Renderer
from lib.colony_helpers import StateBounds
from lib.spawner import BoidSpawner

sb = StateBounds(
    map_dimensions=np.array([50,50], dtype=np.float64),
    min_velocity=0,
    max_velocity=10,
    max_accleration=5,
    max_angular_velocity=np.pi*0.5,
    num_leaders=3,
    num_followers=3
)

bs = BoidSpawner(sb)
cs = bs.generateSpawnState()

bc = BoidsColony(
    init_state=cs,
    bounds=sb,
    radius_repulsion=2,
    radius_orientation=3,
    radius_attraction=5,
    repulsion_multiplier=3,
    orientation_multiplier=1,
    attraction_multiplier=1,
    wall_avoidance_multiplier=1,
    dt=1/60
)

r = Renderer(boids_colony=bc, pixels_per_unit=10)

while not r.checkForPygameQuit():
    bc.step()
    r.renderFrame()
    sleep(1/60)
