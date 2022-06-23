import numpy as np
import pygame
import pygame.gfxdraw

class Renderer():
    def __init__(self, num_leaders, num_followers, map_size, pixels_per_unit, radii=None, follower_inds=None, render_leader_observations = False) -> None:
        # Save variables
        self.num_leaders = num_leaders
        self.num_followers = num_followers
        self.total_agents = num_followers + num_leaders
        self.map_size = map_size
        self.pixels_per_unit = pixels_per_unit
        self.radii = radii
        self.follower_inds = self.setupFollowerInd(follower_inds)
        self.render_leader_observations = render_leader_observations

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

    def setupFollowerInd(self, follower_inds):
        if follower_inds is None:
            return list(range(self.num_leaders+self.num_followers))
        else:
            return follower_inds

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
        if self.radii is not None and boid_id in self.follower_inds and boid_id < self.num_followers:
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

    def renderFrame(self, positions, headings, bm = None, observations = None, all_obs_positions = None, possible_agents = None):
        self.screen.fill((255,255,255))
        self.renderBoids(positions, headings)
        if self.render_leader_observations:
            # This is a bit of a messy way of getting leader observations to show up here
            # In the future, consider reworking this so Renderer doesn't access the BoidsManager directly
            self.renderLeaderObservations(bm, observations, all_obs_positions, possible_agents)
        pygame.display.flip()

    def getPixels(self, units):
        return np.round(units * self.pixels_per_unit).astype(int)

    def renderLeaderObservations(self, bm, observations, all_obs_positions, possible_agents):
        for leader_id in range(self.num_leaders):
            # Save the heading of the leader wrt world frame
            leader_heading = bm.headings[leader_id+self.num_followers][0]
            # Save the position of the leader in world frame
            leader_position = bm.positions[leader_id+self.num_followers]
            # Render a circle around the leader showing the leader's observation radius
            leader_pix_position = self.getPixelCoords(leader_position)
            pix_obs_radius = self.getPixels(self.radii[2])
            pygame.gfxdraw.circle(self.screen, leader_pix_position[0], leader_pix_position[1], pix_obs_radius, (0,200,0))

            # Get the pixel coordinate of the observed centroid of boids
            centroid_obs = observations[possible_agents[leader_id]][0:2]
            centroid_distance = centroid_obs[0]
            centroid_angle = centroid_obs[1]

            x_centroid = centroid_distance * np.cos(leader_heading+centroid_angle)
            y_centroid = centroid_distance * np.sin(leader_heading+centroid_angle)
            centroid_unit_coords = np.array([x_centroid, y_centroid]) + leader_position

            centroid_pixel_coords = self.getPixelCoords(centroid_unit_coords)
            pygame.gfxdraw.line(self.screen, leader_pix_position[0], leader_pix_position[1], centroid_pixel_coords[0], centroid_pixel_coords[1], (0,100,0))

            # Render lines from the centers of all followers to the center of the observable centroid
            obs_positions = all_obs_positions[leader_id]
            for obs_position in obs_positions:
                obs_pos_pix = self.getPixelCoords(obs_position)
                pygame.gfxdraw.line(self.screen, centroid_pixel_coords[0], centroid_pixel_coords[1], obs_pos_pix[0], obs_pos_pix[1], (0,100,0))

            # Draw a line that shows the leader's current heading wrt world frame
            x_heading_vector_units = np.cos(leader_heading)
            y_heading_vector_units = np.sin(leader_heading)
            # Get the tip of the heading vector in world frame
            heading_vector_units = np.array([x_heading_vector_units, y_heading_vector_units]) + bm.positions[leader_id+self.num_followers]
            heading_vector_pix = self.getPixelCoords(heading_vector_units)

            pygame.gfxdraw.line(self.screen, leader_pix_position[0], leader_pix_position[1], heading_vector_pix[0], heading_vector_pix[1], (200,0,0))
