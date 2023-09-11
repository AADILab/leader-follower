import numpy as np
from time import sleep

import myaml

from leaderfollower.boids_colony import BoidsColony
from leaderfollower.env_renderer import Renderer
from leaderfollower.colony_helpers import StateBounds
from leaderfollower.boid_spawner import BoidSpawner, BoidSpawnRule, HeadingRule, PositionRule, VelocityRule
from leaderfollower.fitness_calculator import FitnessCalculator
from leaderfollower.poi_colony import POIColony, POI
from leaderfollower.poi_spawner import POISpawner, POISpawnRule
from leaderfollower.env_observations import ObservationManager, ObservationRule, SensorType
from leaderfollower.math_helpers import calculateDeltaHeading, calculateDistance

config = myaml.safe_load("configs/default.yaml")

map_dimensions=np.array([config["map_dimensions"]["x"],config["map_dimensions"]["y"]], dtype=np.float64)

print(*config["StateBounds"])

sb = StateBounds(
    map_dimensions=map_dimensions,
    **config["StateBounds"]
)

bs = BoidSpawner(
    bounds=sb,
    **config["BoidSpawner"]
)

cs = bs.getSpawnState()

bc = BoidsColony(
    init_state=cs,
    bounds=sb,
    **config["BoidsColony"]
)

ps = POISpawner(
    map_dimensions=map_dimensions,
    **config["POISpawner"]
)

sp = ps.getSpawnPositions()

pc = POIColony(
    positions=sp,
    **config["POIColony"]
)

fc = FitnessCalculator(poi_colony=pc, boids_colony=bc)

om = ObservationManager(
    boids_colony=bc,
    poi_colony=pc,
    **config["ObservationManager"]
)

r = Renderer(boids_colony=bc,
    poi_colony=pc,
    observation_manager=om,
    **config["Renderer"]
)

while not r.checkForPygameQuit():
    bc.step()
    pc.updatePois(bc.state)
    r.renderFrame()
    sleep(1/60)
