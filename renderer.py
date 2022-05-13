import numpy as np
import pygame

class Renderer():
    def __init__(self, num_leaders, num_followers, map_size, pixels_per_unit) -> None:
        # Save variables
        self.num_leaders = num_leaders
        self.num_followers = num_followers
        self.total_agents = num_followers + num_leaders
        self.map_size = map_size
        self.pixels_per_unit = pixels_per_unit

        # Set useful variables
        self.follower_color = (0,0,255)
        self.leader_color = (250, 120, 0)
        self.boid_radius = 1 # unit, not pixel
        self.boid_pixel_radius = self.getPixels(self.boid_radius)
        self.phi = np.pi/8

        # Calculate display variables
        self.display_size = self.getPixels(map_size)

        # Initialize pygame display
        pygame.init()
        self.screen = pygame.display.set_mode(self.display_size)

    def getPixelCoords(self, unit_coords):
        if len(unit_coords.shape) == 1:
            px = self.getPixels(unit_coords[0])
            py = self.getPixels(self.map_size[1] - unit_coords[1])
            return np.array([px, py])
        else:
            p = np.zeros(unit_coords.shape)
            p[:,0] = self.getPixels(unit_coords[:, 0])
            print(self.map_size[1])
            print( - unit_coords[:, 1])
            p[:,1] = self.getPixels(self.map_size[1] - unit_coords[:, 1])
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
        # print(R.shape, points.shape)
        def rotateFunc(point):
            return point.dot(R)
        return np.apply_along_axis(rotateFunc, 1, points)

    def translatePoints(self, points, translation_vec):
        """Translate points by input translation vector"""
        return points + translation_vec

    def generateBoidTrianglePix(self, position, heading):
        pts = self.createTrianglePoints()
        # print(pts.shape)
        print("pts:\n", pts)
        r_pts = self.rotatePoints(pts, heading)
        print("r_pts:\n", r_pts)
        t_pts = self.translatePoints(r_pts, position)
        print("t_pts:\n", t_pts)
        return self.getPixelCoords(t_pts)

    def renderBoids(self, positions, headings):
        for boid_id in range(self.total_agents):
            if boid_id < self.num_followers:
                color = self.follower_color
            else:
                color = self.leader_color
            pixel_coords_2 = self.generateBoidTrianglePix(positions[boid_id], headings[boid_id][0])
            print(pixel_coords_2)
            pygame.draw.polygon(self.screen, color, pixel_coords_2)

    def renderFrame(self, positions, headings):
        self.screen.fill((255,255,255))
        self.renderBoids(positions, headings)
        pygame.display.flip()

    def getPixels(self, units):
        return units * self.pixels_per_unit
