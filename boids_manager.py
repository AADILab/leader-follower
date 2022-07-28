import numpy as np

from map_utils import Map

class BoidsManager():
    def __init__(self, max_velocity, max_angular_velocity, radius_repulsion, radius_orientation, radius_attraction, \
        num_followers, num_leaders, map_size, positions = None, headings = None, velocities = None, \
            avoid_walls = True, ghost_density = 0, use_momentum=False, dt = 1/60, max_acceleration = 5, wall_avoidance_multiplier = 1, repulsion_mulitplier = 3) -> None:
        # Note: Boids are organized in arrays as [followers, leaders]. Followers are at the front of the arrays
        # and Leaders are at the back.
        # Leader index "N" is Boid index "num_followers+N". Follower index "F" is Boid index "F".
        # Note: Boids do not make any distinction between leaders and followers in their observations.
        # In the boid observations, a boid just shows up as a boid whether it's a leader or follower

        # Double check radii are valid
        if radius_repulsion < radius_orientation and radius_orientation < radius_attraction \
            and np.all(np.array([radius_repulsion, radius_orientation, radius_attraction]) > 0):
                pass
        else:
            raise Exception("Double check that radius_repulsion < radius_orientation < radius_attraction. All radii must be > 0.")

        # Save input variables to internal variables
        self.max_velocity = max_velocity
        self.min_velocity = 0
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity
        self.radius_repulsion = radius_repulsion
        self.radius_orientation = radius_orientation
        self.radius_attraction = radius_attraction
        self.total_agents = num_followers + num_leaders
        self.num_followers = num_followers
        self.num_leaders = num_leaders
        self.map_size = map_size
        self.avoid_walls = avoid_walls
        self.wall_avoidance_multiplier = wall_avoidance_multiplier
        self.repulsion_multiplier = repulsion_mulitplier
        self.ghost_density = ghost_density
        self.use_momentum = use_momentum
        self.dt = dt

        # Setup counterfactual ghost boids for wall avoidance
        # This isn't being used for anything right now,
        # but I'm keeping it around in case I need it later.
        self.ghost_positions = self.generate_ghost_positions()
        self.ghost_map = Map(self.map_size, self.radius_repulsion, self.ghost_positions)

        self.init_positions = positions
        self.init_headings = headings
        self.init_velocities = velocities

        # Setup boid positions
        self.positions = self.setup_positions(self.init_positions)

        # Setup boid headings. Headings are represented in map frame.
        self.headings = self.setup_headings(self.init_headings)

        # Setup boid velocities. Velocities range from 0 to max velocity
        self.velocities = self.setup_velocities(self.init_velocities)

        # Setup underlying map structure for observations
        self.map = Map(self.map_size, self.radius_attraction, self.positions)

    def reset(self):
        """Reset simulation state with initial conditions. Random variables will be used for variables where no initial condition was specified."""
        # Setup boid positions
        self.positions = self.setup_positions(self.init_positions)

        # Setup boid headings. Headings are represented in map frame.
        self.headings = self.setup_headings(self.init_headings)

        # Setup boid velocities. Velocities range from 0 to max velocity
        self.velocities = self.setup_velocities(self.init_velocities)

        # Reset the map with the new positions
        self.map.reset(self.positions)

        return None

    def generate_ghost_positions(self):
        """Generate ghost positions along the edges of the map, spaced out by the ghost density."""
        # Simplify variables for indexing
        M, N = self.map_size
        i, j = (M*self.ghost_density)+1, (N*self.ghost_density)-1
        num_ghosts = 2*i + 2*j
        ghost_positions = np.zeros((num_ghosts, 2))
        # Populate ghosts along top and bottom edge
        ghost_positions[:i,0] = np.linspace(0,M,i,endpoint=True)
        ghost_positions[i:2*i,0] = ghost_positions[:i,0]
        ghost_positions[i:2*i,1] = N
        # Populate ghosts along left and right edge
        ghost_positions[2*i:2*i+j,1] = np.linspace(1/self.ghost_density,N-1/self.ghost_density,j,endpoint=True)
        ghost_positions[2*i+j:,1] = ghost_positions[2*i:2*i+j,1]
        ghost_positions[2*i+j:,0] = M
        return ghost_positions

    def setup_velocities(self, velocities):
        if velocities is None:
            # Boid velocities are randomized from min to the max velocity
            return np.random.uniform(self.min_velocity, self.max_velocity, size=(self.total_agents,1))
        else:
            if type(velocities) != np.ndarray:
                raise Exception("velocities must be input as numpy array")
            elif velocities.shape != (self.total_agents, 1):
                raise Exception("velocities must be shape (total agents, 1)")
            else:
                return velocities.copy()

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
                return headings.copy()

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
                return positions.copy()

    def get_observable_boid_ids(self, boid_id):
        # Grab the observable boids in that position
        observable_boid_ids = self.map.get_observable_agent_inds(self.positions[boid_id], self.positions)
        # Toss out the current boid from observable boids
        observable_boid_ids.remove(boid_id)
        return observable_boid_ids

    def get_follower_positions(self):
        return self.positions[:self.num_followers]

    def get_leader_positions(self):
        """Get positions of leader agents"""
        return self.positions[self.num_followers:]

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

    def get_rep_ghost_positions(self, position):
        # This gets all ghost positions within the radius of repulsion of the input position
        obs_ghost_inds = self.ghost_map.get_observable_agent_inds(position, self.ghost_positions)
        return self.ghost_positions[obs_ghost_inds]

    @staticmethod
    def calculate_centroid(positions):
        if positions.size == 0:
            return None
        else:
            return np.average(positions, axis=0)

    def get_leader_position_observations(self):
        """Give the absolute x,y for each observable boid in leader's observation radius."""
        all_obs_positions = []
        # For every leader
        for boid_id in np.arange(self.num_leaders)+self.num_followers:
            # Get observable boid ids
            obs_boid_ids = self.get_observable_boid_ids(boid_id)
            # Get observable boid positions
            obs_positions = self.get_boid_positions(obs_boid_ids)
            # Store the position
            all_obs_positions.append(obs_positions)
        # Return observable positions for all leaders
        return all_obs_positions

    def get_leader_centroid_observations(self):
        """Centroid observations are a 2d array organized as [distance, angle] for each leader"""
        centroids_obs_np = np.zeros((self.num_leaders,2))
        # For every leader
        for boid_id in np.arange(self.num_leaders)+self.num_followers:
            # Get observable boid ids
            obs_boid_ids = self.get_observable_boid_ids(boid_id)
            # Get observable boid positions
            obs_positions = self.get_boid_positions(obs_boid_ids)
            # Calculate centroid of observable boids. Return own position if there are no observable boids.
            centroid = self.calculate_centroid(obs_positions)
            if centroid is not None:
                # centroid relative to leader boid
                relative_centroid = centroid - self.positions[boid_id]
                # Calculate distance to centroid
                distance = np.linalg.norm(relative_centroid)
                # Get leader heading
                leader_heading = self.headings[boid_id]
                # Get angle from leader to centroid in world frame
                centroid_angle = np.arctan2(relative_centroid[1], relative_centroid[0])
                # Calculate angle from leader heading to centroid
                angle = self.bound_heading_pi_to_pi(centroid_angle - leader_heading)
            else:
                # There are no observable boids.
                # Create abritrarily large distance.
                # Create abritrary middle angle.
                distance = 1000
                angle = 0
            # Save distance and angle as observation for that leader
            centroids_obs_np[boid_id-self.num_followers,0] = distance
            centroids_obs_np[boid_id-self.num_followers,1] = angle
        return centroids_obs_np

    @staticmethod
    def bound_heading_pi_to_pi(heading):
        bounded_heading = heading
        # Bound heading from [0,2pi]
        if bounded_heading > 2*np.pi or bounded_heading < 0:
            bounded_heading %= 2*np.pi
        # Bound heading from [-pi,+pi]
        if bounded_heading > np.pi:
            bounded_heading -= 2*np.pi
        return bounded_heading

    def get_leader_relative_position_observations(self, positions):
        """Get relative heading and distance to specified positions for all leaders."""
        all_leader_obs = []
        for leader_ind in range(self.num_leaders):
            leader_obs = np.zeros((positions.shape[0], 2))
            for num_position, position in enumerate(positions):
                # Calculate relative position from leader to position
                relative_position = position - self.positions[self.num_followers+leader_ind]
                # Calculate distance to position
                distance = np.linalg.norm(relative_position)
                # Get leader heading
                leader_heading = self.headings[self.num_followers+leader_ind][0]
                # Calculate angle to position in world frame
                angle = np.arctan2(relative_position[1], relative_position[0])
                # Get relative angle to position bounded from -pi to +pi
                relative_angle = self.bound_heading_pi_to_pi(angle - leader_heading)
                leader_obs[num_position] = [distance, relative_angle]
            # Save this leader's observation to all leaders' observations
            all_leader_obs.append(leader_obs)
        return all_leader_obs

    def get_leader_distance_to_position(self, position):
        return np.linalg.norm(self.positions[self.num_followers:] - position, axis=0)

    def get_leader_distance_to_positions(self, positions):
        all_distances = np.zeros((self.num_leaders*positions.shape[0],2))
        for position in positions:
            all_distances[self.num_followers:] = self.get_leader_distance_to_position(position)
        return all_distances

    def get_follower_observations(self):
        all_obs_rep_boids_pos = []      # all observable repulsion boid positions
        all_obs_orient_boids_head = []   # all observable orientation boid headings
        all_obs_attract_boids_pos = []  # all observable attraction boid positions
        no_boid_obs_inds = [] # indices of boids which have no observations
        for boid_id in range(self.num_followers):
            # Get observable boid ids
            obs_boid_ids = self.get_observable_boid_ids(boid_id)
            # Get positions and headings of observable boids
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
            if len(rep_positions) == 0 and len(orient_headings) == 0 and len(attract_positions) == 0:
                no_boid_obs_inds.append(boid_id)

        return all_obs_rep_boids_pos, all_obs_orient_boids_head, all_obs_attract_boids_pos, no_boid_obs_inds

    def calculate_repulsion_vector(self, boid_id, repulsion_positions):
        # Repulsion vector is average vector from repulsion boids to current boid, normalized by radius of repulsion
        if np.shape(repulsion_positions)[0] != 0:
            return (self.positions[boid_id] - np.average(repulsion_positions, axis=0))/self.radius_repulsion * self.repulsion_multiplier
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

    def get_no_forces_ind(self, no_boids_obs_inds, wall_avoidance_inds):
        no_forces_ind = []
        for id in no_boids_obs_inds:
            if id not in wall_avoidance_inds:
                no_forces_ind.append(id)
        return no_forces_ind

    def calculate_all_repulsion_vectors(self, all_obs_rep_boids_pos):
        all_repulsion_vectors = np.zeros((self.num_followers, 2))
        for boid_id, rep_boids_pos in enumerate(all_obs_rep_boids_pos[:self.num_followers]):
            all_repulsion_vectors[boid_id] = self.calculate_repulsion_vector(boid_id, rep_boids_pos)
        return all_repulsion_vectors

    def calculate_all_orientation_vectors(self, all_obs_orient_boids_head):
        all_orientation_vectors = np.zeros((self.num_followers, 2))
        for boid_id, orient_boid_pos in enumerate(all_obs_orient_boids_head[:self.num_followers]):
            all_orientation_vectors[boid_id] = self.calculate_orientation_vector(orient_boid_pos)
        return all_orientation_vectors

    def calculate_all_attraction_vectors(self, all_obs_attract_boids_pos):
        all_attraction_vectors = np.zeros((self.num_followers, 2))
        for boid_id, attract_boid_pos, in enumerate(all_obs_attract_boids_pos[:self.num_followers]):
            all_attraction_vectors[boid_id] = self.calculate_attraction_vector(boid_id, attract_boid_pos)
        return all_attraction_vectors

    def calculate_all_wall_avoidance_vectors(self):
        # Initialize wall avoidance vectors at zero
        all_wall_avoidance_vectors = np.zeros((self.num_followers, 2))

        # Get indicies of boids near bottom edge
        bool_bottom = self.positions[:self.num_followers,1] <= self.radius_repulsion
        # Set up wall avoidance vectors for bottom edge
        all_wall_avoidance_vectors[:,1][bool_bottom] = self.radius_repulsion - self.positions[:self.num_followers][:,1][bool_bottom]

        # Get indices of boids near left edge
        bool_left = self.positions[:self.num_followers,0] <= self.radius_repulsion
        # Set up wall avoidanve vectors for left edge
        all_wall_avoidance_vectors[:,0][bool_left] = self.radius_repulsion - self.positions[:self.num_followers][:,0][bool_left]

        # Get indices of boids near top edge
        bool_top = self.positions[:self.num_followers,1] >= self.map_size[1] - self.radius_repulsion
        # Set up wall avoidance vectors for top edge
        all_wall_avoidance_vectors[:,1][bool_top] = self.map_size[1] - self.radius_repulsion - self.positions[:self.num_followers][:,1][bool_top]

        # Get indices of boids near right edge
        bool_right = self.positions[:self.num_followers,0] >= self.map_size[0] - self.radius_repulsion
        # Set up wall avoidance vectors for right edge
        all_wall_avoidance_vectors[:,0][bool_right] = self.map_size[0] - self.radius_repulsion - self.positions[:self.num_followers][:,0][bool_right]

        return all_wall_avoidance_vectors

    def calculate_all_wall_avoidance_inds(self, all_wall_avoidance_vectors):
        inds = []
        wall_sum = np.abs(all_wall_avoidance_vectors[:,0]) + np.abs(all_wall_avoidance_vectors[:,1])
        for ind, ws in enumerate(wall_sum):
            if ws > 0:
                inds.append(ind)
        return inds

    def calculate_follower_desired_states(self, all_obs_rep_boids_pos, all_obs_orient_boids_head, all_obs_attract_boids_pos, no_boid_obs_inds, debug=False):
        # Calculate repulsion vectors for all follower boids
        all_repulsion_vectors = self.calculate_all_repulsion_vectors(all_obs_rep_boids_pos)

        # Calculate orientation vectors for all follower boids
        all_orientation_vectors = self.calculate_all_orientation_vectors(all_obs_orient_boids_head)

        # Calculate attraction vectors for all follower boids
        all_attraction_vectors = self.calculate_all_attraction_vectors(all_obs_attract_boids_pos)

        # Calculate wall avoidance vector if wall avoidance is on
        if self.avoid_walls:
            all_wall_avoidance_vectors = self.total_agents * self.calculate_all_wall_avoidance_vectors() * self.wall_avoidance_multiplier
        else:
            all_wall_avoidance_vectors = np.zeros((self.num_followers, 2))

        # Calculate a momentum for follower boids
        if self.use_momentum:
            all_momentum_vectors = np.hstack((
                self.velocities[:self.num_followers]*np.cos(self.headings[:self.num_followers]),
                self.velocities[:self.num_followers]*np.sin(self.headings[:self.num_followers])
            ))
        else:
            all_momentum_vectors = np.zeros((self.num_followers, 2))

        # Calculate desired boid velocities and headings from vector sums
        all_sum_vectors = all_repulsion_vectors + all_orientation_vectors + all_attraction_vectors + all_wall_avoidance_vectors + all_momentum_vectors
        all_desired_headings = np.expand_dims(np.arctan2(all_sum_vectors[:,1], all_sum_vectors[:,0]), axis=1)

        # Calculate desired velocity depending on how aligned the desired and current headings are
        all_delta_headings = self.calculate_delta_headings(all_desired_headings, self.headings[:self.num_followers])
        # all_desired_velocities = np.expand_dims(np.linalg.norm(all_sum_vectors , axis=1), axis=1)
        all_desired_velocities = np.expand_dims(np.zeros(self.num_followers), axis=1)
        all_desired_velocities[np.abs(all_delta_headings) < np.pi/2] = np.expand_dims(np.linalg.norm(all_sum_vectors , axis=1), axis=1)[np.abs(all_delta_headings) < np.pi/2]
        all_desired_velocities[np.abs(all_delta_headings) >= np.pi/2] = self.min_velocity

        # Boids should maintain current heading and velocity if no repulsion, orientation, attraction, or
        # wall avoidance vectors are acting on them
        no_forces_ind = self.get_no_forces_ind(no_boid_obs_inds, self.calculate_all_wall_avoidance_inds(all_wall_avoidance_vectors))
        all_desired_headings[no_forces_ind] = self.headings[no_forces_ind]
        all_desired_velocities[no_forces_ind] = self.velocities[no_forces_ind]

        # Return all the data calculated if we're interested in debugging results of in-between steps
        if debug:
            return all_desired_headings, all_desired_velocities, all_sum_vectors, \
                all_repulsion_vectors, all_orientation_vectors, all_attraction_vectors

        return all_desired_headings, all_desired_velocities

    def calculate_delta_headings(self, desired_headings, current_headings):
        """ Calculate delta headings such that delta is the shortest path from
        current heading to the desired heading. Ensure delta headings fit within
        maximum angular velocity bounds.
        """
        num_headings = current_headings.shape[0]
        delta_headings = np.zeros((num_headings, 1))
        for heading_id in range(num_headings):
            desired_heading = desired_headings[heading_id, 0]
            current_heading = current_headings[heading_id, 0]
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
            delta_headings[heading_id, 0] = delta_heading

        return delta_headings

    def calculate_delta_velocities(self, all_desired_velocities, all_current_velocities):
        return all_desired_velocities - all_current_velocities

    def calculate_follower_deltas(self, follower_desired_headings, follower_desired_velocities):
        """Calculate delta headings and delta velocities based on current follower
        headings and velocities and their DESIRED headings and velocities. Do not apply
        any bounding logic."""
        delta_headings = self.calculate_delta_headings(follower_desired_headings, self.headings[:self.num_followers])
        delta_velocities = self.calculate_delta_velocities(follower_desired_velocities, self.velocities[:self.num_followers])
        return delta_headings, delta_velocities

    def calculate_follower_kinematics(self, delta_headings, delta_velocities):
        """Turn deltas for heading and velocity into angular velocities
        and linear acclerations. Bound kinematics according to specified
        boundaries. Ex: max acceleration, max angular velocity
        """
        angular_velocities = delta_headings/self.dt
        angular_velocities[angular_velocities > self.max_angular_velocity] = self.max_angular_velocity
        angular_velocities[angular_velocities < -self.max_angular_velocity] = -self.max_angular_velocity
        accelerations = delta_velocities/self.dt
        accelerations[accelerations > self.max_acceleration] = self.max_acceleration
        accelerations[accelerations < -self.max_acceleration] = -self.max_acceleration
        return angular_velocities, accelerations

    def update_all_states(self, angular_velocities, accelerations):
        """Update all leader and follower states with the input kinematics using Euler integration.
        Bound kinematics according to specified boundaries. Ex: max_velocity
        """
        # Update headings
        self.headings += angular_velocities*self.dt
        # Apply circular cutoff
        self.headings %= (2*np.pi)
        # Update velocities
        self.velocities += accelerations*self.dt
        self.velocities[self.velocities > self.max_velocity] = self.max_velocity
        self.velocities[self.velocities < self.min_velocity] = self.min_velocity
        # Update positions
        self.positions[:,0] += self.velocities[:,0] * np.cos(self.headings[:,0]) * self.dt
        self.positions[:,1] += self.velocities[:,0] * np.sin(self.headings[:,0]) * self.dt
        # Bound positions
        self.bound_positions()
        return None

    def update_follower_states(self, angular_velocities, accelerations):
        """Update follower states with the input kinematics using Euler integration.
        Bound kinematics according to specified boundaries. Ex: max velocity
        """
        # Update headings
        self.headings[:self.num_followers] += angular_velocities*self.dt
        # Apply circular cutoff
        self.headings[:self.num_followers] %= (2*np.pi)
        # Update velocities
        self.velocities[:self.num_followers] += accelerations*self.dt
        self.velocities[:self.num_followers][self.velocities[:self.num_followers] > self.max_velocity] = self.max_velocity
        self.velocities[:self.num_followers][self.velocities[:self.num_followers] < self.min_velocity] = self.min_velocity
        # Update positions
        self.positions[:self.num_followers][:,0] += self.velocities[:self.num_followers][:,0] * np.cos(self.headings[:self.num_followers][:,0]) * self.dt
        self.positions[:self.num_followers][:,1] += self.velocities[:self.num_followers][:,0] * np.sin(self.headings[:self.num_followers][:,0]) * self.dt
        # Bound positions
        self.bound_positions()
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

    def unpack_leader_actions(self, leader_actions):
        if leader_actions is None:
            return np.zeros((self.num_leaders,1)), self.velocities[self.num_followers:]
        else:
            return np.expand_dims(leader_actions[:,0], axis=1), np.expand_dims(leader_actions[:,1], axis=1)

    def get_leader_velocities(self):
        return self.velocities[self.num_followers:]

    def step(self, leader_actions=None):
        """Step forward the simulation
        leader_actions is Nx2. Left side is delta headings. Right side is desired velocities.
        If leader actions is None, then leaders remain in current state.
        """
        # Unpack leader actions
        leader_delta_headings, leader_desired_velocities = self.unpack_leader_actions(leader_actions)
        # Update the follower observations
        repulsion_boids, orientation_boids, attraction_boids, no_boid_obs_inds = self.get_follower_observations()
        # Update follower desired states
        follower_desired_headings, follower_desired_velocities = self.calculate_follower_desired_states(repulsion_boids, orientation_boids, attraction_boids, no_boid_obs_inds)
        # Calculate follower delta states
        follower_delta_headings, follower_delta_velocities = self.calculate_follower_deltas(follower_desired_headings, follower_desired_velocities)
        # Calculate leader delta velocities
        leader_delta_velocities = self.calculate_delta_velocities(leader_desired_velocities, self.get_leader_velocities())
        # Package together follower delta states and leader delta states
        all_delta_headings = np.vstack((follower_delta_headings, leader_delta_headings))
        all_delta_velocities = np.vstack((follower_delta_velocities, leader_delta_velocities))
        # Turn delta states into kinematics commands
        angular_velocities, accelerations = self.calculate_follower_kinematics(all_delta_headings, all_delta_velocities)
        # Update leader and follower states using kinematics
        self.update_all_states(angular_velocities, accelerations)
        # Reset the map with the new positions
        self.map.reset(self.positions)
