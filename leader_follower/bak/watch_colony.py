# todo
# from time import sleep
#
# import numpy as np
#
# from lib.boid_spawner import BoidSpawner, BoidSpawnRule
# from lib.boids_colony import BoidsColony
# from lib.colony_helpers import StateBounds
# from lib.fitness_calculator import FitnessCalculator
# from lib.observations_manager import ObservationManager, ObservationRule, SensorType
# from lib.poi_colony import POIColony
# from lib.poi_spawner import POISpawner, POISpawnRule
# from lib.renderer import Renderer
#
# map_dimensions = np.array([100, 100], dtype=np.float64)
#
# sb = StateBounds(
#     map_dimensions=map_dimensions,
#     min_velocity=0,
#     max_velocity=10,
#     max_acceleration=5,
#     max_angular_velocity=np.pi * 0.5,
#     num_leaders=5,
#     num_followers=40
# )
#
# bs = BoidSpawner(
#     bounds=sb,
#     spawn_rule=BoidSpawnRule.Circle,
#     # position_rule=PositionRule.CenterCircle,
#     # velocity_rule=VelocityRule.FixedStart,
#     # heading_rule=HeadingRule.UniformRandom,
#     radius_fraction=1 / 5,
#     velocity_fraction=1 / 2,
# )
#
# cs = bs.getSpawnState()
#
# bc = BoidsColony(
#     init_state=cs,
#     bounds=sb,
#     radius_repulsion=2,
#     radius_orientation=3,
#     radius_attraction=5,
#     repulsion_multiplier=3,
#     orientation_multiplier=1,
#     attraction_multiplier=1,
#     wall_avoidance_multiplier=1,
#     dt=1 / 60
# )
#
# ps = POISpawner(
#     poi_spawn_rule=POISpawnRule.BoundedRandom,
#     num_pois=10,
#     map_dimensions=map_dimensions,
#     bound_fraction=0.5
# )
#
# sp = ps.getSpawnPositions()
#
# pc = POIColony(
#     positions=sp,
#     observation_radius=5,
#     coupling=5
# )
#
# fc = FitnessCalculator(poi_colony=pc, boids_colony=bc, fitness=FitnessCalculator.difference_no_leaders_followers)
#
# om = ObservationManager(
#     observation_rule=ObservationRule.Individual,
#     boids_colony=bc,
#     poi_colony=pc,
#     num_poi_bins=4,
#     num_swarm_bins=12,
#     poi_sensor_type=SensorType.InverseDistance,
#     swarm_sensor_type=SensorType.InverseDistance
# )
#
# r = Renderer(boids_colony=bc, poi_colony=pc, observation_manager=om, pixels_per_unit=10)
#
# while not r.checkForPygameQuit():
#     bc.step(leader_desired_velocities=0 * np.ones(sb.num_leaders),
#             leader_desired_delta_headings=-np.pi * np.ones(sb.num_leaders))
#     pc.updatePois(bc.state)
#     r.renderFrame()
#     sleep(1 / 60)
