from typing import Tuple, List

import numpy as np
import pygame
import pygame.gfxdraw
from numpy.typing import NDArray

from leader_follower.boids_colony import BoidsColony, Boid
from leader_follower.observations_manager import ObservationManager
from leader_follower.poi_colony import POIColony, POI


class Renderer:
    def __init__(self, boids_colony: BoidsColony, poi_colony: POIColony, observation_manager: ObservationManager,
                 pixels_per_unit: int, leader_colors: List[Tuple[int]]):
        # Save variables
        self.boids_colony = boids_colony
        self.poi_colony = poi_colony
        self.observation_manager = observation_manager
        self.pixels_per_unit = pixels_per_unit

        # Setup colors
        self.follower_color = (0, 120, 250)
        # self.leader_colors = [
        #     (250, 120, 0),
        #     (250, 250, 0),
        #     (0, 200, 0),
        #     (120,0,120),
        #     (66,245,242)
        # ]
        self.leader_colors = leader_colors

        self.poi_observed_color = (0, 255, 0)
        self.poi_not_observed_color = (255, 0, 0)

        self.boid_radius = 1  # unit, not pixel
        self.boid_pix_radius = self.get_pixels(self.boid_radius)
        self.phi = np.pi / 8  # angle for boid triangle

        # Set up pygame and display
        pygame.init()
        self.display_size = self.get_pixels(self.boids_colony.bounds.map_dimensions)
        self.screen = pygame.display.set_mode(self.display_size)
        return

    def get_pixels(self, units):
        return np.round(units * self.pixels_per_unit).astype(int)

    def get_pixel_coords(self, unit_coords):
        if len(unit_coords.shape) == 1:
            px = self.get_pixels(unit_coords[0])
            py = self.get_pixels(self.boids_colony.bounds.map_dimensions[1] - unit_coords[1])
            return np.array([px, py])
        else:
            p = np.zeros(unit_coords.shape, dtype=int)
            p[:, 0] = self.get_pixels(unit_coords[:, 0])
            p[:, 1] = self.get_pixels(self.boids_colony.bounds.map_dimensions[1] - unit_coords[:, 1])
            return p

    def create_triangle_points(self):
        """Generates points for boid triangle centered at the origin w. heading=0"""
        return np.array([
            [1 / 2, 0],
            [-1 / 2, np.tan(self.phi)],
            [-1 / 2, -np.tan(self.phi)]
        ])

    @staticmethod
    def rotate_points(points, angle):
        """Rotates points around the origin by input angle"""
        r = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        def rotate(point):
            return r.dot(point.T).T

        return np.apply_along_axis(rotate, 1, points)

    @staticmethod
    def translate_points(points, translation_vec):
        """Translate points by input translation vector"""
        return points + translation_vec

    def gen_boid_triangle_pix(self, position, heading):
        pts = self.create_triangle_points()
        r_pts = self.rotate_points(pts, heading)
        t_pts = self.translate_points(r_pts, position)
        return self.get_pixel_coords(t_pts)

    def boid_color(self, boid: Boid):
        # This color assignment will have to change if boids can change between
        # being a leader vs follower. For now, this works.
        if boid.is_leader():
            return self.leader_colors[boid.id % len(self.leader_colors)]
        else:
            return self.follower_color

    def render_boid(self, boid: Boid):
        pix_coords = self.gen_boid_triangle_pix(boid.position, boid.heading)
        color = self.boid_color(boid)
        pygame.gfxdraw.aapolygon(self.screen, pix_coords, color)
        pygame.gfxdraw.filled_polygon(self.screen, pix_coords, color)
        return

    def render_boids(self):
        for boid in self.boids_colony.boids:
            self.render_boid(boid)
        return

    def render_poi(self, poi: POI):
        pix_coords = self.get_pixel_coords(poi.position)
        if poi.observed:
            color = self.poi_observed_color
        else:
            color = self.poi_not_observed_color
        pygame.gfxdraw.aacircle(self.screen, pix_coords[0], pix_coords[1], int(self.pixels_per_unit / 4), color)
        pygame.gfxdraw.filled_circle(self.screen, pix_coords[0], pix_coords[1], int(self.pixels_per_unit / 4), color)
        pygame.gfxdraw.aacircle(self.screen, pix_coords[0], pix_coords[1],
                                int(self.poi_colony.observation_radius * self.pixels_per_unit), color)
        return

    def render_pois(self):
        for poi in self.poi_colony.pois:
            self.render_poi(poi)
        return

    def render_sensor_readings(self, leader: Boid, sensor_readings: NDArray[np.float64], color,
                               leader_pix_coords: NDArray[np.int64]):
        for ind, sensor_reading in enumerate(sensor_readings):
            # Angle for this sensor reading relative to leader heading
            angle_segment = 2 * np.pi / (2 * sensor_readings.size)
            relative_angle = -np.pi + angle_segment + angle_segment * 2 * ind
            # Angle in world from leader to this sensor reading
            absolute_angle = (leader.heading + relative_angle)  # % (2*np.pi)
            # Position of reading to display
            relative_position = np.array(
                [sensor_reading * np.cos(absolute_angle), sensor_reading * np.sin(absolute_angle)])
            # Position in world frame
            absolute_position = leader.position + relative_position
            # Convert to pix
            poi_pix_coords = self.get_pixel_coords(absolute_position)
            # Render line
            pygame.gfxdraw.line(self.screen, leader_pix_coords[0], leader_pix_coords[1], poi_pix_coords[0],
                                poi_pix_coords[1], color)
        return

    def render_observation(self, observation: NDArray[np.float64], leader: Boid):
        leader_pix_coords = self.get_pixel_coords(leader.position)
        self.render_sensor_readings(
            leader, observation[:self.observation_manager.num_poi_bins], (0, 0, 0), leader_pix_coords
        )
        self.render_sensor_readings(
            leader, observation[self.observation_manager.num_poi_bins:], self.boid_color(leader), leader_pix_coords
        )
        return

    def render_observations(self):
        for observation, leader in zip(self.observation_manager.get_all_observations(), self.boids_colony.leaders()):
            self.render_observation(observation, leader)
        return

    def render_frame(self):
        self.screen.fill((255, 255, 255))
        self.render_observations()
        self.render_boids()
        self.render_pois()
        pygame.display.flip()
        return

    @staticmethod
    def check_for_pygame_quit():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
