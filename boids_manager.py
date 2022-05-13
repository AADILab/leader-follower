import numpy as np

from map_utils import Map

class BoidsManager():
    def __init__(self, max_velocity, max_angular_velocity, radius_repulsion, radius_orientation, radius_attraction, \
        num_followers, num_leaders, map_size, positions = None, headings = None) -> None:
        # Note: Boids are organized in arrays as [followers, leaders]. Followers are at the front of the arrays
        # and Leaders are at the back.
        # Leader index "N" is Boid index "num_followers+N". Follower index "F" is Boid index "F".

        # Double check radii are valid
        if radius_repulsion < radius_orientation and radius_orientation < radius_attraction \
            and np.all(np.array([radius_repulsion, radius_orientation, radius_attraction]) > 0):
                pass
        else:
            raise Exception("Double check that radius_repulsion < radius_orientation < radius_attraction. All radii must be > 0.")

        # Save input variables to internal variables
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
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
            # Boid headings are randomized from 0 to 2π.
            return np.random.uniform(0, 2*np.pi, size=(self.total_agents,1))
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

    def get_follower_observations(self):
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

    def calculate_repulsion_vector(self, boid_id, repulsion_positions):
        # Repulsion vector is average vector from repulsion boids to current boid, normalized by radius of repulsion
        if np.shape(repulsion_positions)[0] != 0:
            return (self.positions[boid_id] - np.average(repulsion_positions, axis=0))/self.radius_repulsion
        else:
            return np.array([0,0])

    def calculate_orientation_vector(self, orientation_headings):
        # Orientation vector is sum of vectors derived from orientations of orientation boids
        if np.shape(orientation_headings)[0] != 0:
            # Calculate a unit (x,y) vector from each heading
            unit_vectors = np.hstack((
                np.cos(orientation_headings),
                np.sin(orientation_headings)
            ))
            # Sum up the vectors for the final orientation vector
            return np.sum(unit_vectors, axis=0)
        else:
            return np.array([0,0])

    def calculate_attraction_vector(self, boid_id, attract_boid_pos):
        # Attraction vector is average vector from current boid to attraction boids, normalized by radius of attraction
        if np.shape(attract_boid_pos)[0] != 0:
            return (np.average(attract_boid_pos, axis=0) - self.positions[boid_id])/self.radius_attraction
        else:
            return np.array([0,0])

    def calculate_follower_desired_actions(self, all_obs_rep_boids_pos, all_obs_orient_boids_head, all_obs_attract_boids_pos, debug=False):
        # Calculate repulsion vectors for all follower boids
        all_repulsion_vectors = np.zeros((self.num_followers, 2))
        for boid_id, rep_boids_pos in enumerate(all_obs_rep_boids_pos[:self.num_followers]):
            all_repulsion_vectors[boid_id] = self.calculate_repulsion_vector(boid_id, rep_boids_pos)

        # Calculate orientation vectors for all follower boids
        all_orientation_vectors = np.zeros((self.num_followers, 2))
        for boid_id, orient_boid_pos in enumerate(all_obs_orient_boids_head[:self.num_followers]):
            all_orientation_vectors[boid_id] = self.calculate_orientation_vector(orient_boid_pos)

        # Calculate attraction vectors for all follower boids
        all_attraction_vectors = np.zeros((self.num_followers, 2))
        for boid_id, attract_boid_pos, in enumerate(all_obs_attract_boids_pos[:self.num_followers]):
            all_attraction_vectors[boid_id] = self.calculate_attraction_vector(boid_id, attract_boid_pos)

        # Calculate desired boid velocities and headings from vector sums
        all_sum_vectors = all_repulsion_vectors + all_orientation_vectors + all_attraction_vectors
        all_desired_headings = np.expand_dims(np.arctan2(all_sum_vectors[:,1], all_sum_vectors[:,0]), axis=1)
        all_desired_velocities = np.expand_dims(np.linalg.norm(all_sum_vectors , axis=1), axis=1)

        # Return all the data calculated if we're interested in de ugging results of in-between steps
        if debug:
            return all_desired_headings, all_desired_velocities, all_sum_vectors, \
                all_repulsion_vectors, all_orientation_vectors, all_attraction_vectors

        return all_desired_headings, all_desired_velocities

    def calculate_delta_headings(self, all_desired_headings):
        """ Calculate delta headings such that delta is the shortest path from
        current heading to the desired heading. Ensure delta headings fit within
        maximum angular velocity bounds.
        """
        delta_headings = np.zeros((self.num_followers, 1))
        for boid_id in range(self.num_followers):
            desired_heading = all_desired_headings[boid_id, 0]
            current_heading = self.headings[boid_id, 0]
            # Case 0: Desired heading is current heading
            if desired_heading == current_heading:
                delta_heading = 0
            else:
                # Case 1: Desired heading greater than current heading
                if desired_heading > current_heading:
                    desired_heading_prime = desired_heading - 2*np.pi

                # Case 2: Desired heading less than current heading
                else:
                    desired_heading_prime = desired_heading + 2*np.pi

                delta0 = desired_heading - current_heading
                delta1 = desired_heading_prime - current_heading
                which_delta = np.argmin([np.abs(delta0), np.abs(delta1)])
                delta_heading = np.array([delta0, delta1])[which_delta]
            # Save calculated delta heading
            delta_headings[boid_id, 0] = delta_heading

        # Bound delta headings according to max angular velocity constraint
        delta_headings[delta_headings > self.max_angular_velocity] = self.max_angular_velocity
        delta_headings[delta_headings < -self.max_angular_velocity] = -self.max_angular_velocity

        return delta_headings

    def update_follower_states(self, all_desired_headings, all_desired_velocities):
        # Bound velocities
        self.bound_velocities(all_desired_velocities)
        # Calculate delta applied to headings
        delta_headings = self.calculate_delta_headings(all_desired_headings)
        # Apply that delta to headings
        self.headings[:self.num_followers] += delta_headings
        # Apply circular cutoff to headings
        self.headings[:self.num_followers] %= (2*np.pi)
        # Apply velocities to new positions
        self.positions[:self.num_followers][:,0] += all_desired_velocities[:,0] * np.cos(self.headings[:self.num_followers][:,0])
        self.positions[:self.num_followers][:,1] += all_desired_velocities[:,0] * np.sin(self.headings[:self.num_followers][:,0])
        return None

    def bound_velocities(self, velocities):
        # Bounds velocities as an in-place operation
        over_max_bool = velocities > self.max_velocity
        velocities[over_max_bool] = self.max_velocity
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
        repulsion_boids, orientation_boids, attraction_boids = self.get_follower_observations()
        # Update the actions
        desired_headings, velocities = self.calculate_follower_desired_actions(repulsion_boids, orientation_boids, attraction_boids)
        # Apply actions
        self.update_follower_states(desired_headings, velocities)
        # Apply boundary conditions on positions
        self.bound_positions()
        # Reset the map with the new positions
        self.map.reset(self.positions)
