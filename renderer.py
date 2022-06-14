import numpy as np
import pygame
import pygame.gfxdraw

class Renderer():
    def __init__(self, num_leaders, num_followers, map_size, pixels_per_unit, radii=None, r_ind=None) -> None:
        # Save variables
        self.num_leaders = num_leaders
        self.num_followers = num_followers
        self.total_agents = num_followers + num_leaders
        self.map_size = map_size
        self.pixels_per_unit = pixels_per_unit
        self.radii = radii
        self.r_ind = self.setupRInd(r_ind)

        # Set useful variables
        self.follower_color = (0,120,250)
        self.leader_color = (250, 120, 0)
        self.boid_radius = 1 # unit, not pixel
        self.boid_pixel_radius = self.getPixels(self.boid_radius)
        self.phi = np.pi/8

        # Calculate display variables
        self.display_size = self.getPixels(map_size)

        # Initialize pygame display
        pygame.init()
        self.screen = pygame.display.set_mode(self.display_size)

    def setupRInd(self, r_ind):
        if r_ind is None:
            return list(range(self.num_leaders+self.num_followers))
        else:
            return r_ind

    def getPixelCoords(self, unit_coords):
        if len(unit_coords.shape) == 1:
            px = self.getPixels(unit_coords[0])
            py = self.getPixels(self.map_size[1] - unit_coords[1])
            return np.array([px, py])
        else:
            p = np.zeros(unit_coords.shape)
            p[:,0] = self.getPixels(unit_coords[:, 0])
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

    def renderBoid(self, position, heading, color, boid_id):
        pix_coords = self.generateBoidTrianglePix(position, heading)
        pygame.gfxdraw.aapolygon(self.screen, pix_coords, color)
        pygame.gfxdraw.filled_polygon(self.screen, pix_coords, color)
        if self.radii is not None and boid_id in self.r_ind:
            center_pix_coord = self.getPixelCoords(position).astype(int)
            self.renderCircle(center_pix_coord, self.getPixels(self.radii[0]), (100,0,0))
            self.renderCircle(center_pix_coord, self.getPixels(self.radii[1]), (150,0,0))
            self.renderCircle(center_pix_coord, self.getPixels(self.radii[2]), (200,0,0))

    def renderCircle(self, center_pix_coord, pix_radius, color):
        pygame.gfxdraw.aacircle(self.screen, center_pix_coord[0], center_pix_coord[1], pix_radius, color)

    def renderBoids(self, positions, headings):
        for boid_id in range(self.total_agents):
            if boid_id < self.num_followers:
                color = self.follower_color
            else:
                color = self.leader_color
            self.renderBoid(positions[boid_id], headings[boid_id][0], color, boid_id)

    def renderFrame(self, positions, headings):
        self.screen.fill((255,255,255))
        self.renderBoids(positions, headings)
        pygame.display.flip()

    def getPixels(self, units):
        return np.round(units * self.pixels_per_unit).astype(int)
