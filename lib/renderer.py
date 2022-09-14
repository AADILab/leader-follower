from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import pygame
import pygame.gfxdraw

from lib.boids_colony import BoidsColony, Boid
from lib.poi_colony import POIColony, POI
from lib.observations_manager import ObservationManager, SensorType

class Renderer():
    def __init__(self, boids_colony: BoidsColony, poi_colony: POIColony, observation_manager: ObservationManager, pixels_per_unit: int) -> None:
        # Save variables
        self.boids_colony = boids_colony
        self.poi_colony = poi_colony
        self.observation_manager = observation_manager
        self.pixels_per_unit = pixels_per_unit

        # Setup colors
        self.follower_color = (0,120,250)
        self.leader_colors = [
            (250, 120, 0),
            (250, 250, 0),
            (0, 200, 0),
            (120,0,120),
            (66,245,242)
        ]

        self.poi_observed_color = (0,255,0)
        self.poi_not_observed_color = (255,0,0)

        self.boid_radius = 1 # unit, not pixel
        self.boid_pix_radius = self.getPixels(self.boid_radius)
        self.phi = np.pi/8 # angle for boid triangle

        # Set up pygame and display
        pygame.init()
        self.display_size = self.getPixels(self.boids_colony.bounds.map_dimensions)
        self.screen = pygame.display.set_mode(self.display_size)

    def getPixels(self, units):
        return np.round(units * self.pixels_per_unit).astype(int)

    def getPixelCoords(self, unit_coords):
        if len(unit_coords.shape) == 1:
            px = self.getPixels(unit_coords[0])
            py = self.getPixels(self.boids_colony.bounds.map_dimensions[1] - unit_coords[1])
            return np.array([px, py])
        else:
            p = np.zeros(unit_coords.shape, dtype=int)
            p[:,0] = self.getPixels(unit_coords[:, 0])
            p[:,1] = self.getPixels(self.boids_colony.bounds.map_dimensions[1] - unit_coords[:, 1])
            return p

    def createTrianglePoints(self):
        """Generates points for boid triangle centered at the origin w. heading=0"""
        return np.array([
            [1/2, 0],
            [-1/2, np.tan(self.phi)],
            [-1/2, -np.tan(self.phi)]
        ])

    def rotatePoints(self, points, angle):
        """Rotates points around the origin by input angle"""
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        def rotateFunc(point):
            return R.dot(point.T).T
        return np.apply_along_axis(rotateFunc, 1, points)

    def translatePoints(self, points, translation_vec):
        """Translate points by input translation vector"""
        return points + translation_vec

    def generateBoidTrianglePix(self, position, heading):
        pts = self.createTrianglePoints()
        r_pts = self.rotatePoints(pts, heading)
        t_pts = self.translatePoints(r_pts, position)
        return self.getPixelCoords(t_pts)

    def boidColor(self, boid: Boid):
        # This color assignment will have to change if boids can change between
        # being a leader vs follower. For now, this works.
        if boid.isLeader():
            return self.leader_colors[boid.id%len(self.leader_colors)]
        else:
            return self.follower_color

    def renderBoid(self, boid: Boid):
        pix_coords = self.generateBoidTrianglePix(boid.position, boid.heading)
        color = self.boidColor(boid)
        pygame.gfxdraw.aapolygon(self.screen, pix_coords, color)
        pygame.gfxdraw.filled_polygon(self.screen, pix_coords, color)

    def renderBoids(self):
        # print("P: ", self.boids_colony.boids[0].position)
        for boid in self.boids_colony.boids:
            self.renderBoid(boid)

    def renderPOI(self, poi: POI):
        pix_coords = self.getPixelCoords(poi.position)
        if poi.observed:
            color = self.poi_observed_color
        else:
            color = self.poi_not_observed_color
        pygame.gfxdraw.aacircle(self.screen, pix_coords[0], pix_coords[1], int(self.pixels_per_unit/4), color)
        pygame.gfxdraw.filled_circle(self.screen, pix_coords[0], pix_coords[1], int(self.pixels_per_unit/4), color)
        pygame.gfxdraw.aacircle(self.screen, pix_coords[0], pix_coords[1], int(self.poi_colony.observation_radius * self.pixels_per_unit), color)

    def renderPOIs(self):
        for poi in self.poi_colony.pois:
            self.renderPOI(poi)

    def renderSensorReadings(self, leader: Boid, sensor_readings: NDArray[np.float64], color: Tuple[float], leader_pix_coords: NDArray[np.int64]):
        for ind, sensor_reading in enumerate(sensor_readings):
            # Angle for this sensor reading relative to leader heading
            angle_segment = 2*np.pi / (2*sensor_readings.size)
            relative_angle = -np.pi + angle_segment + angle_segment*2*ind
            # Angle in world from leader to this sensor reading
            absolute_angle = (leader.heading + relative_angle)# % (2*np.pi)
            # Position of reading to display
            relative_position = np.array([sensor_reading*np.cos(absolute_angle), sensor_reading*np.sin(absolute_angle)])
            # Position in world frame
            absolute_position = leader.position+relative_position
            # Convert to pix
            poi_pix_coords = self.getPixelCoords(absolute_position)
            # Render line
            pygame.gfxdraw.line(self.screen, leader_pix_coords[0], leader_pix_coords[1], poi_pix_coords[0], poi_pix_coords[1], color)

    def renderObservation(self, observation: NDArray[np.float64], leader: Boid):
        leader_pix_coords = self.getPixelCoords(leader.position)
        self.renderSensorReadings(leader, observation[:self.observation_manager.num_poi_bins], (0,0,0), leader_pix_coords)
        self.renderSensorReadings(leader, observation[self.observation_manager.num_poi_bins:], self.boidColor(leader), leader_pix_coords)

    def renderObservations(self):
        for observation, leader in zip(self.observation_manager.getAllObservations(), self.boids_colony.getLeaders()):
            self.renderObservation(observation, leader)

    def renderFrame(self):
        self.screen.fill((255,255,255))
        self.renderObservations()
        self.renderBoids()
        self.renderPOIs()
        pygame.display.flip()

    @staticmethod
    def checkForPygameQuit():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
