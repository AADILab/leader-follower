import numpy as np

from map_utils import Map

class BoidsManager():
    def __init__(self, max_velocity, angular_velocity, radius_repulsion, radius_orientation, radius_attraction, \
        num_followers, num_leaders, map_size, positions = None, headings = None, velocities = None) -> None:
        # Note: Boids are organized in arrays as [followers, leaders]. Followers are at the front of the arrays
        # and Leaders are at the back. Leader index "N" is Boid index "num_followers+N"

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
        self.total_agents = num_followers + num_leaders
        self.num_followers = num_followers
        self.num_leaders = num_leaders
        self.map_size = map_size

        # Setup boid positions
        self.positions = self.setup_positions(positions)

        # Setup boid headings. Headings are represented in map frame.
        self.headings = self.setup_headings(headings)

        # Setup boid velocities. Velocities are 1D, greater than 0
        self.velocities = self.setup_velocities(velocities)

        # Setup underlying map structure for observations
        self.map = Map(self.map_size, self.radius_attraction, self.positions)

    def setup_velocities(self, velocities):
        if velocities is None:
            # Boid velocities are randomized from 0 to the max velocity
            return np.random.uniform(0, self.max_velocity, size=(self.total_agents,1))
        else:
            if type(velocities) != np.ndarray:
                raise Exception("velocities must be input as numpy array")
            elif velocities.shape != (self.total_agents, 1):
                raise Exception("velocities must be shape (total agents, 1)")
            else:
                return velocities

    def setup_headings(self, headings):
        if headings is None:
            # Boid headings are randomized from -π to +π.
            return np.random.uniform(-np.pi, np.pi, size=(self.total_agents,1))
        else:
            if type(headings) != np.ndarray:
                raise Exception("headings must be input as numpy array")
            elif headings.shape != (self.total_agents, 1):
                raise Exception("headings must be shape (total agents, 1)")
            else:
                return headings

    def setup_positions(self, positions):
        if positions is None:
            # Boid positions are randomized within map bounds
            # The first num_leaders boids are leaders, and there rest are regular boids
            return np.hstack((
                np.random.uniform(self.map_size[0], size=(self.total_agents,1)),
                np.random.uniform(self.map_size[1], size=(self.total_agents,1))
            ))
        else:
            if type(positions) != np.ndarray:
                raise Exception("positions must be input as numpy array")
            elif positions.shape != (self.total_agents, 2):
                raise Exception("positions must be shape (total agents, 1)")
            else:
                return positions

    def get_observable_boid_ids(self, boid_id):
        # Grab the observable boids in that position
        observable_boid_ids = self.map.get_observable_agent_inds(self.positions[boid_id], self.positions)
        # Toss out the current boid from observable boids
        observable_boid_ids.remove(boid_id)
        return observable_boid_ids

    def get_boid_positions(self, ids):
        return self.positions[ids]

    def get_boid_headings(self, ids):
        return self.headings[ids]

    def get_distances(self, positions, boid_id):
        # Returns distances from input positions to boid with specified boid id
        return np.linalg.norm(positions - self.positions[boid_id], axis=1)

    def get_repulsion_boid_positions(self, observable_distances, observable_positions):
        in_repulsion_radius_bool = observable_distances <= self.radius_repulsion
        return observable_positions[in_repulsion_radius_bool]

    def get_orientation_boid_headings(self, observable_distances, observable_headings):
        out_repulsion_radius_bool = observable_distances > self.radius_repulsion
        in_orientation_radius_bool = observable_distances <= self.radius_orientation
        orientation_bool = np.logical_and(in_orientation_radius_bool, out_repulsion_radius_bool)
        return observable_headings[orientation_bool]

    def get_attraction_boid_positions(self, observable_distances, observable_positions):
        out_orientation_radius_bool = observable_distances > self.radius_orientation
        in_attraction_radius_bool = observable_distances <= self.radius_attraction
        attraction_bool = np.logical_and(in_attraction_radius_bool, out_orientation_radius_bool)
        return observable_positions[attraction_bool]

    def update_follower_observations(self):
        all_obs_rep_boids_pos = []      # all observable repulsion boid positions
        all_obs_orient_boids_head = []   # all observable orientation boid headings
        all_obs_attract_boids_pos = []  # all observable attraction boid positions
        for boid_id in range(self.num_followers):
            # Get observable boid ids
            obs_boid_ids = self.get_observable_boid_ids(boid_id)
            # Get positions of observable boids
            obs_positions = self.get_boid_positions(obs_boid_ids)
            obs_headings = self.get_boid_headings(obs_boid_ids)
            # Get distance of observable boids to current boid
            obs_distances = self.get_distances(obs_positions, boid_id)
            # Get observable boid positions within repulsion radius
            rep_positions = self.get_repulsion_boid_positions(obs_distances, obs_positions)
            # Get observable boid positions within orientation radius and outside repulsion radius
            orient_headings = self.get_orientation_boid_headings(obs_distances, obs_headings)
            # Get observable boid positions within attraction radius and outside orientation radius
            attract_positions = self.get_attraction_boid_positions(obs_distances, obs_positions)
            # Save to overall lists
            all_obs_rep_boids_pos.append(rep_positions)
            all_obs_orient_boids_head.append(orient_headings)
            all_obs_attract_boids_pos.append(attract_positions)

        return all_obs_rep_boids_pos, all_obs_orient_boids_head, all_obs_attract_boids_pos

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
