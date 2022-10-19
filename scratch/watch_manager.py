from lib.boids_manager import BoidsManager
from lib.renderer import Renderer

import numpy as np
from time import sleep

# min_velocity=0 by default in bm

bm = BoidsManager(
    positions=np.array([
        [2,2],
        [3,3],
        [2,3]
    ], dtype=float),
    headings=np.array([[np.pi/2], [np.pi/3], [-np.pi]], dtype=float),
    velocities=np.array([[0],[0],[0]], dtype=float),
    radius_repulsion=2,
    radius_orientation=3,
    radius_attraction=5,
    repulsion_mulitplier=3,
    map_size=np.array([50,50], dtype=int),
    max_velocity=10,
    max_acceleration=5,
    max_angular_velocity=np.pi*0.5,
    dt=1/60,
    num_followers=3,
    num_leaders=0,
    ghost_density=10
)

# bc = BoidsColony(
#     leader_positions=[],
#     follower_positions=[
#         [20,20],
#         [21,20],
#         [22,20]
#     ],
#     leader_headings=[],
#     follower_headings=[np.pi/2, np.pi/3, -np.pi],
#     leader_velocities=[],
#     follower_velocities=[0,0,0],
#     radius_repulsion=2,
#     radius_orientation=3,
#     radius_attraction=5,
#     repulsion_mulitplier=3,
#     map_dimensions=[50,50],
#     min_velocity=0, max_velocity=10,
#     max_acceleration=5,
#     max_angular_velocity=np.pi*0.5,
#     dt=1/60
# )

# class Renderer():
#     def __init__(self, num_leaders, num_followers, map_size, pixels_per_unit, radii=None, follower_inds=None, render_mode = 'human') -> None:

r = Renderer(num_leaders=0, num_followers=3, map_size=np.array([50,50],dtype=int), pixels_per_unit=10)
    # def renderFrame(self, bm: BoidsManager, pm: POIManager = None,
    # env_observations: Dict = None, all_obs_positions = None, possible_agents = None,
    # render_POIs: bool = True, render_centroid_observations: bool = True):
while not r.checkForPygameQuit():
# for _ in range(3):
    bm.step(leader_actions=None)
    r.renderFrame(bm, pm=None, env_observations=None, all_obs_positions=None, possible_agents=None, render_POIs=False, render_centroid_observations=False)
    sleep(1/60)
