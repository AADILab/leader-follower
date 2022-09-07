import numpy as np
from time import sleep

from lib.boids_colony import BoidsColony, Boid
from lib.env_renderer import Renderer

bc = BoidsColony(
    leader_positions=[],
    follower_positions=[
        [20,20],
        [21,20],
        [22,20]
    ],
    leader_headings=[],
    follower_headings=[np.pi/2, np.pi/3, -np.pi],
    leader_velocities=[],
    follower_velocities=[0,0,0],
    radius_repulsion=2,
    radius_orientation=3,
    radius_attraction=5,
    repulsion_mulitplier=3,
    map_dimensions=[50,50],
    min_velocity=0, max_velocity=10,
    max_acceleration=5,
    max_angular_velocity=np.pi*0.5,
    dt=1/60
)

r = Renderer(boids_colony=bc, pixels_per_unit=10)

while not r.checkForPygameQuit():
# for _ in range(60):
    bc.step(np.array([]), np.array([]))
    r.renderFrame()
    sleep(1/60)
