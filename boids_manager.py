import numpy as np

from map_utils import Map

class BoidsManager():
    def __init__(self, max_velocity, angular_velocity, radius_repulsion, radius_orientation, radius_attraction, \
        num_followers, num_leaders, map_size, positions = None, headings = None, velocities = None) -> None:
        # Double check radii are valid
        if radius_repulsion < radius_orientation and radius_orientation < radius_attraction \
            and np.all(np.array([radius_repulsion, radius_orientation, radius_attraction]) > 0):
                pass
        else:
            raise Exception("Double check that radius_repulsion < radius_orientation < radius_attraction. All radii must be > 0.")

        # Save input variables to internal variables
        self.max_velocity = max_velocity
        self.angular_velocity = angular_velocity
        self.radius_repulsion = radius_repulsion
        self.radius_orientation = radius_orientation
        self.radius_attraction = radius_attraction
        self.total_agents = num_leaders + num_followers
        self.num_leaders = num_leaders
        self.num_followers = num_followers

        # Setup boid positions
        if positions is None:
            # Boid positions are randomized within map bounds
            # The first num_leaders boids are leaders, and there rest are regular boids
            self.positions = np.hstack((
                np.random.uniform(map_size[0], size=(self.total_agents,1)),
                np.random.uniform(map_size[1], size=(self.total_agents,1))
            ))
        else:
            if type(positions) != np.ndarray:
                raise Exception("positions must be input as numpy array")
            elif positions.shape != (self.total_agents, 2):
                raise Exception("positions must be shape (total agents, 1)")
            else:
                self.positions = positions

        # Setup boid headings. Headings are represented in map frame.
        if headings is None:
            # Boid headings are randomized from -π to +π.
            self.headings = np.random.uniform(-np.pi, np.pi, size=(self.total_agents,1))
        else:
            if type(headings) != np.ndarray:
                raise Exception("headings must be input as numpy array")
            elif headings.shape != (self.total_agents, 1):
                raise Exception("headings must be shape (total agents, 1)")
            else:
                self.headings = headings

        # Setup boid velocities. Velocities are 1D, greater than 0
        if velocities is None:
            self.velocities = np.random.uniform(0, self.max_velocity, size=(self.total_agents,1))
        else:
            if type(velocities) != np.ndarray:
                raise Exception("velocities must be input as numpy array")
            elif velocities.shape != (self.total_agents, 1):
                raise Exception("velocities must be shape (total agents, 1)")
            else:
                self.velocities = velocities

        # Setup underlying map structure for observations
        self.map = Map(map_size, self.radius_attraction, self.positions)

    def update_follower_observations(self):
        repulsion_boids = []
        orientation_boids = []
        attraction_boids = []
        for ind, position in enumerate(self.positions[self.num_leaders:]):
            # Apply offset to id based on how many leaders are in the sim
            boid_id = self.num_leaders+ind
            # Grab the observable boids in that position
            observable_boid_ids = self.map.get_observable_agent_inds(position, self.positions)
            # Toss out the current boid from observable boids
            observable_boid_ids.remove(boid_id)
            # Get positions and headings of observable boids
            observable_boid_positions = self.positions[observable_boid_ids]
            observable_boid_headings = self.headings[observable_boid_ids]
            # Get distance of observable boids to current boid
            observable_boid_distances = np.linalg.norm(observable_boid_positions - position, axis=1)
            # Get all observable boid positions within repulsion radius
            in_repulsion_radius_bool = observable_boid_distances <= self.radius_repulsion
            repulsion_boids_positions = observable_boid_positions[in_repulsion_radius_bool]
            # Get all observable boid positions inside orientation radius and not inside repulsion radius
            in_orientation_radius_bool = observable_boid_distances <= self.radius_orientation
            in_orientation_out_repulsion = np.logical_and(in_orientation_radius_bool, np.logical_not(in_repulsion_radius_bool))
            orientation_boids_headings = observable_boid_headings[in_orientation_out_repulsion]
            # Get all observable boid positions inside attraction radius and not inside orientation radius
            in_attraction_radius_bool = observable_boid_distances <= self.radius_attraction
            in_attraction_out_orientation = np.logical_and(in_attraction_radius_bool, np.logical_not(in_orientation_radius_bool))
            attraction_boids_positions = observable_boid_positions[in_attraction_out_orientation]
            # Save to overall lists
            repulsion_boids.append(repulsion_boids_positions)
            orientation_boids.append(orientation_boids_headings)
            attraction_boids.append(attraction_boids_positions)

        return repulsion_boids, orientation_boids, attraction_boids

    def update_follower_actions(self, repulsion_boids, orientation_boids, attraction_boids):
        desired_headings = np.zeros((self.num_followers,1))
        velocities = np.zeros((self.num_followers,1))
        for ind, (position, repulsion_positions, orientation_headings, attraction_positions) \
            in enumerate(zip(self.positions[self.num_leaders:], repulsion_boids[self.num_leaders:], \
                orientation_boids[self.num_leaders:], attraction_boids[self.num_leaders:])):
            # Apply offset to id based on how many leaders are in the sim
            boid_id = self.num_leaders+ind
            # Calculate repulsion vector
            if np.shape(repulsion_positions)[0] != 0:
                repulsion_vector = position - np.average(repulsion_positions, axis=0)
            else:
                repulsion_vector = np.array([0,0])
            # Calculate orientation vector
            if np.shape(orientation_headings)[0] != 0:
                avg_orientation = np.average(orientation_headings)
                orientation_vector = np.array([ np.cos(avg_orientation), np.sin(avg_orientation)])
            else:
                orientation_vector = np.array([0,0])
            # Calculate attraction vector
            if np.shape(attraction_positions)[0] != 0:
                attraction_vector = np.average(attraction_positions, axis=0) - position
            else:
                attraction_vector = np.array([0,0])
            # Sum vectors influencing boid to determine boid velocity and desired heading
            total_vector = repulsion_vector+orientation_vector+attraction_vector
            desired_heading = np.arctan2(total_vector[0], total_vector[1])
            velocity = np.linalg.norm(total_vector)
            # Add velocities and desired headings to lists for tracking all follower actions
            desired_headings[boid_id] = desired_heading
            if velocity > self.max_velocity:
                velocities[boid_id] = self.max_velocity
            else:
                velocities[boid_id] = velocity

        return desired_headings, velocities

    def apply_follower_actions(self, desired_headings, velocities):
        # Calculate delta applied to headings
        delta_headings = desired_headings - self.headings[self.num_leaders:]
        delta_headings[delta_headings>self.angular_velocity] = self.angular_velocity
        # Apply that delta to headings
        self.headings[self.num_leaders:] += delta_headings
        # Apply circular cutoff to headings
        self.headings[self.num_leaders:] %= (2*np.pi)
        # Apply velocities to new positions
        self.positions[self.num_leaders:] += velocities
        return None

    def bound_positions(self):
        # Apply left bound
        self.positions[:,0][self.positions[:,0]<0] = 0
        # Apply right bound
        self.positions[:,0][self.positions[:,0]>self.map.map_size[0]] = self.map.map_size[0]
        # Apply lower bound
        self.positions[:,1][self.positions[:,1]<0] = 0
        # Apply upper bound
        self.positions[:,1][self.positions[:,1]>self.map.map_size[1]] = self.map.map_size[1]

    def step(self):
        # Update the observations
        repulsion_boids, orientation_boids, attraction_boids = self.update_follower_observations()
        # Update the actions
        desired_headings, velocities = self.update_follower_actions(repulsion_boids, orientation_boids, attraction_boids)
        # Apply actions
        self.apply_follower_actions(desired_headings, velocities)
        # Apply boundary conditions on positions
        self.bound_positions()
        # Reset the map with the new positions
        self.map.reset(self.positions)
