import numpy as np
import pygame
import pygame.gfxdraw

from lib.boids_colony import BoidsColony, Boid

class Renderer():
    def __init__(self, boids_colony: BoidsColony, pixels_per_unit: int) -> None:
        # Save variables
        self.boids_colony = boids_colony
        self.pixels_per_unit = pixels_per_unit

        # Setup colors
        self.follower_color = (0,120,250)
        self.leader_colors = [
            (250, 120, 0),
            (250, 250, 0),
            (0, 200, 0),
            (120,0,120)
        ]

        self.poi_observed_color = (0,255,0)
        self.poi_not_observed_color = (255,0,0)

        self.boid_radius = 1 # unit, not pixel
        self.boid_pix_radius = self.getPixels(self.boid_radius)
        self.phi = np.pi/8 # angle for boid triangle

        # Set up pygame and display
        pygame.init()
        self.display_size = self.getPixels(self.boids_colony.map_dimensions)
        self.screen = pygame.display.set_mode(self.display_size)

    def getPixels(self, units):
        return np.round(units * self.pixels_per_unit).astype(int)

    def getPixelCoords(self, unit_coords):
        if len(unit_coords.shape) == 1:
            px = self.getPixels(unit_coords[0])
            py = self.getPixels(self.boids_colony.map_dimensions[1] - unit_coords[1])
            return np.array([px, py])
        else:
            p = np.zeros(unit_coords.shape, dtype=int)
            p[:,0] = self.getPixels(unit_coords[:, 0])
            p[:,1] = self.getPixels(self.boids_colony.map_dimensions[1] - unit_coords[:, 1])
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
        for boid in self.boids_colony.boids:
            self.renderBoid(boid)

    def renderFrame(self):
        self.screen.fill((255,255,255))
        self.renderBoids()
        pygame.display.flip()

    @staticmethod
    def checkForPygameQuit():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
