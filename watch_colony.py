import numpy as np
from time import sleep

from lib.boids_colony import BoidsColony
from lib.env_renderer import Renderer
from lib.colony_helpers import StateBounds
from lib.boid_spawner import BoidSpawner, BoidSpawnRule, HeadingRule, PositionRule, VelocityRule
from lib.poi_colony import POIColony, POI
from lib.poi_spawner import POISpawner, POISpawnRule
from lib.env_observations import ObservationManager, ObservationRule, SensorType
from lib.math_helpers import calculateDeltaHeading, calculateDistance

map_dimensions=np.array([100,100], dtype=np.float64)

sb = StateBounds(
    map_dimensions=map_dimensions,
    min_velocity=0,
    max_velocity=10,
    max_accleration=5,
    max_angular_velocity=np.pi*0.5,
    num_leaders=10,
    num_followers=50
)

bs = BoidSpawner(
    bounds=sb,
    spawn_rule=BoidSpawnRule.Circle,
    # position_rule=PositionRule.CenterCircle,
    # velocity_rule=VelocityRule.FixedStart,
    # heading_rule=HeadingRule.UniformRandom,
    radius_fraction=1/5,
    velocity_fraction=1/2,
)

cs = bs.getSpawnState()

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

ps = POISpawner(
    poi_spawn_rule=POISpawnRule.BoundedRandom,
    num_pois = 5,
    map_dimensions=map_dimensions,
    bound_fraction=0.5
)

sp = ps.getSpawnPositions()

pc = POIColony(
    positions=sp,
    observation_radius=5,
    coupling=5
)

def updatePois():
    for poi in pc.pois:
        distances = calculateDistance(poi.position, bc.state.positions)
        num_observations = np.sum(distances<=pc.observation_radius)
        if num_observations >= pc.coupling:
            poi.observed = True
            # Get ids of swarm members that observed this poi
            observer_ids = np.nonzero(distances<=pc.observation_radius)[0]
            poi.observation_list.append(observer_ids)

om = ObservationManager(
    observation_rule=ObservationRule.Individual,
    boids_colony=bc,
    poi_colony=pc,
    num_poi_bins=4,
    num_swarm_bins=8,
    poi_sensor_type=SensorType.InverseDistance,
    swarm_sensor_type=SensorType.InverseDistance
)

r = Renderer(boids_colony=bc, poi_colony=pc, observation_manager=om, pixels_per_unit=10)

while not r.checkForPygameQuit():
    bc.step()
    updatePois()
    r.renderFrame()
    sleep(1/60)
