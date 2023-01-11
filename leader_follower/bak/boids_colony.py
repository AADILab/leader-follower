from typing import Optional

import numpy as np
from numpy.typing import NDArray

from leader_follower.bak import colony_helpers, math_helpers


class Boid:
    def __init__(self, colony_state: colony_helpers.BoidsColonyState,
                 state_bounds: colony_helpers.StateBounds, boid_id: int):
        self.colony_state = colony_state
        self.id = boid_id
        if not self.is_leader():
            # Index is leader. Value is how many timesteps that leader was within the
            # observation radius of this follower
            self.leader_influence = [0 for _ in range(state_bounds.num_leaders)]

    @property
    def position(self) -> np.ndarray:
        return self.colony_state.positions[self.id]

    @property
    def heading(self) -> np.float64:
        return self.colony_state.headings[self.id]

    @property
    def velocity(self) -> np.float64:
        return self.colony_state.velocities[self.id]

    def is_leader(self) -> bool:
        return self.colony_state.is_leader[self.id]


BoidArray = np.ndarray[Boid, np.dtype[Boid]]


class BoidsColony:
    def __init__(self,
                 init_state: colony_helpers.BoidsColonyState, bounds: colony_helpers.StateBounds,
                 radius_repulsion: float, radius_orientation: float, radius_attraction: float,
                 repulsion_multiplier: float, orientation_multiplier: float, attraction_multiplier: float,
                 wall_avoidance_multiplier: float, dt: float
                 ) -> None:

        self.state = init_state
        self.bounds = bounds

        self.boids = np.array([Boid(self.state, self.bounds, boid_id) for boid_id in range(self.bounds.num_total)])

        self.radius_repulsion = radius_repulsion
        self.radius_orientation = radius_orientation
        self.radius_attraction = radius_attraction

        self.repulsion_multiplier = repulsion_multiplier
        self.orientation_multiplier = orientation_multiplier
        self.attraction_multiplier = attraction_multiplier
        self.wall_avoidance_multiplier = wall_avoidance_multiplier

        self.dt = dt
        return

    def reset(self, reset_state: colony_helpers.BoidsColonyState) -> None:
        self.state.__dict__.update(reset_state.__dict__)
        return

    def leaders(self) -> BoidArray:
        """Get all the leaders in an array"""
        return self.boids[:self.bounds.num_leaders]

    def followers(self) -> BoidArray:
        """Get all the followers in an array"""
        return self.boids[self.bounds.num_leaders:]

    def observable_boids(self, boid: Boid, return_distances: bool = False):
        """Get all boids observable by this boid"""
        distances = math_helpers.euclidean(self.state.positions, boid.position)
        observable_bool = distances <= self.radius_attraction
        observable_bool[boid.id] = False
        # todo fix return hint
        if return_distances:
            return self.boids[observable_bool], distances[observable_bool]
        return self.boids[observable_bool]

    def split_rep_ori_att(self, observable_boids: BoidArray, distances: NDArray[np.float64]):
        """Split observable boids into repulsion, orientation, and attraction boids"""
        repulsion_bool = distances <= self.radius_repulsion
        orientation_bool = np.logical_and(distances > self.radius_repulsion, distances <= self.radius_orientation)
        attraction_bool = np.logical_and(distances > self.radius_orientation, distances <= self.radius_attraction)

        repulsion_boids = observable_boids[repulsion_bool]
        orientation_boids = observable_boids[orientation_bool]
        attraction_boids = observable_boids[attraction_bool]

        return repulsion_boids, orientation_boids, attraction_boids

    def repulsion_vector(self, boid: Boid, repulsion_boids: BoidArray) -> NDArray[np.float64]:
        """Calculate the repulsion vector for this boid using the repulsion_boids as the boids to move away from"""
        # Repulsion vector is average vector from repulsion boids to current boid, normalized by radius of repulsion
        repulsion_positions = np.array([boid.position for boid in repulsion_boids])
        if np.shape(repulsion_positions)[0] != 0:
            return (boid.position - np.average(repulsion_positions,
                                               axis=0)) / self.radius_repulsion * self.repulsion_multiplier
        else:
            return np.array([0., 0.], dtype=np.float64)

    def orientation_vector(self, orientation_boids: BoidArray) -> NDArray[np.float64]:
        """Calculate the orientation vector using the orientation_boids headings as the orientation to match"""
        # Orientation vector is sum of vectors derived from orientations of orientation boids
        orientation_headings = np.array([boid.heading for boid in orientation_boids])
        if np.shape(orientation_headings)[0] != 0:
            # Calculate a unit (x,y) vector from each heading
            unit_vectors = np.hstack((
                np.expand_dims(np.cos(orientation_headings), axis=1),
                np.expand_dims(np.sin(orientation_headings), axis=1)
            ))
            # Sum up the vectors for the final orientation vector
            return np.sum(unit_vectors, axis=0) * self.orientation_multiplier
        else:
            return np.array([0., 0.], dtype=np.float64)

    def attraction_vector(self, boid: Boid, attraction_boids: BoidArray) -> NDArray[np.float64]:
        """Calculate the attraction vector for this boid using the attraction_boids as the boids to move towards"""
        # Attraction vector is average vector from current boid to attraction boids, normalized by radius of attraction
        attraction_positions = np.array([boid.position for boid in attraction_boids])
        if np.shape(attraction_positions)[0] != 0:
            return (np.average(attraction_positions,
                               axis=0) - boid.position) / self.radius_attraction * self.attraction_multiplier
        else:
            return np.array([0., 0.], dtype=np.float64)

    def near_wall(self, boid: Boid) -> bool:
        """Determine if a boid is within repulsion radius of any wall"""
        if np.any(boid.position <= self.radius_repulsion):
            return True
        elif np.any(boid.position > self.bounds.map_dimensions - self.radius_repulsion):
            return True
        else:
            return False

    def near_observable_boids(self, boid: Boid) -> bool:
        """Determine if this boid is close enough to any other boids to observe them"""
        return len(self.observable_boids(boid)) == 0

    def wall_avoidance_vector(self, boid: Boid) -> NDArray[np.float64]:
        """Calculate the wall avoidance vector for this boid using the internal map dimensions"""
        wall_vec = np.array([0, 0], dtype=float)
        # Left wall
        if boid.position[0] <= self.radius_repulsion:
            wall_vec[0] = self.radius_repulsion - boid.position[0]
        # Right wall
        elif boid.position[0] >= self.bounds.map_dimensions[0] - self.radius_repulsion:
            wall_vec[0] = self.bounds.map_dimensions[0] - self.radius_repulsion - boid.position[0]
        # Bottom wall
        if boid.position[1] <= self.radius_repulsion:
            wall_vec[1] = self.radius_repulsion - boid.position[1]
        # Top wall
        elif boid.position[1] >= self.bounds.map_dimensions[1] - self.radius_repulsion:
            wall_vec[1] = self.bounds.map_dimensions[1] - self.radius_repulsion - boid.position[1]
        return wall_vec * self.wall_avoidance_multiplier

    def desired_velocity(self, sum_vector: NDArray[np.float64], delta_heading: float) -> float:
        """Calculate the desired velocity for this boid using the given X,Y sum vector and desired change in heading"""
        # Heading is well aligned
        if np.abs(delta_heading) < np.pi / 2:
            # Match desired velocity to magnitude of sum vector
            return np.linalg.norm(sum_vector)
        # Heading is not well aligned
        else:
            # Slow down
            return self.bounds.min_velocity

    def kinematics(self, delta_velocities: NDArray[np.float64], delta_headings: NDArray[np.float64]):
        """Turn deltas for heading and velocity into angular velocities
        and linear accelerations. Bound kinematics according to specified
        boundaries. Ex: max acceleration, max angular velocity
        """
        angular_velocities = delta_headings / self.dt
        angular_velocities[angular_velocities > self.bounds.max_angular_velocity] = self.bounds.max_angular_velocity
        angular_velocities[angular_velocities < -self.bounds.max_angular_velocity] = -self.bounds.max_angular_velocity
        linear_accelerations = delta_velocities / self.dt
        linear_accelerations[linear_accelerations > self.bounds.max_acceleration] = self.bounds.max_acceleration
        linear_accelerations[linear_accelerations < -self.bounds.max_acceleration] = -self.bounds.max_acceleration
        return angular_velocities, linear_accelerations

    def apply_kinematics(self, angular_velocities: NDArray[np.float64],
                         linear_accelerations: NDArray[np.float64]) -> None:
        """Update all positions, velocities, and headings with the input kinematics using Euler integration.
        Bound kinematics according to specified boundaries. Ex: max_velocity
        """
        # Update headings
        self.state.headings += angular_velocities * self.dt
        # Apply circular cutoff
        self.state.headings %= (2 * np.pi)
        # Update velocities
        self.state.velocities += linear_accelerations * self.dt
        self.state.velocities[self.state.velocities > self.bounds.max_velocity] = self.bounds.max_velocity
        self.state.velocities[self.state.velocities < self.bounds.min_velocity] = self.bounds.min_velocity
        # Update positions
        self.state.positions[:, 0] += self.state.velocities * np.cos(self.state.headings) * self.dt
        self.state.positions[:, 1] += self.state.velocities * np.sin(self.state.headings) * self.dt
        # Bound positions
        # Apply left bound
        self.state.positions[:, 0][self.state.positions[:, 0] < 0] = 0
        # Apply right bound
        self.state.positions[:, 0][self.state.positions[:, 0] > self.bounds.map_dimensions[0]] = \
            self.bounds.map_dimensions[0]
        # Apply lower bound
        self.state.positions[:, 1][self.state.positions[:, 1] < 0] = 0
        # Apply upper bound
        self.state.positions[:, 1][self.state.positions[:, 1] > self.bounds.map_dimensions[1]] = \
            self.bounds.map_dimensions[1]
        return

    def update_leader_influence(self):
        for follower in self.followers():
            observable_boids = self.observable_boids(follower)
            for boid in observable_boids:
                if boid.is_leader():
                    follower.leader_influence[boid.id] += 1

    def step(self, leader_desired_velocities: Optional[NDArray[np.float64]] = None,
             leader_desired_delta_headings: Optional[NDArray[np.float64]] = None) -> None:
        """Step forward the boid colony with the input leader actions"""
        # Update which leader each follower is being influenced by
        self.update_leader_influence()

        # Initialize desired velocities array
        # Initialize desired headings array
        delta_velocities = np.zeros(self.bounds.num_total)
        delta_headings = np.zeros(self.bounds.num_total)

        # Go through each follower
        for follower in self.followers():
            # Get all boids within observation radius
            observable_boids, distances = self.observable_boids(follower, return_distances=True)
            # Determine if boid is near wall
            near_wall = self.near_wall(follower)
            # If no boids are observed, and this boid is not near a wall
            if observable_boids.size == 0 and not near_wall:
                # Calculate its delta velocity as 0.0 and delta heading as 0.0
                delta_velocities[follower.id] = 0.0
                delta_headings[follower.id] = 0.0

            else:
                # Separate observable boids into repulsion, orientation, and attraction boids
                rep_boids, ori_boids, att_boids = self.split_rep_ori_att(observable_boids, distances)
                # Calculate repulsion vector
                repulsion_vec = self.repulsion_vector(follower, rep_boids)
                # Calculate orientation vector
                orientation_vec = self.orientation_vector(ori_boids)
                # Calculate attraction vector
                attraction_vec = self.attraction_vector(follower, att_boids)
                # Calculate wall avoidance vector
                wall_avoid_vec = self.wall_avoidance_vector(follower)
                # Sum vectors together to get x,y vector representing desired trajectory
                sum_vec = repulsion_vec + orientation_vec + attraction_vec + wall_avoid_vec
                # calculate X, Y vector
                # desired heading based on x,y vector
                desired_heading = np.arctan2(sum_vec[1], sum_vec[0])
                # Calculate delta heading required to get desired heading
                delta_heading = math_helpers.get_delta_heading(follower.heading, desired_heading)
                # Calculate desired velocity based on alignment between current heading and desired heading
                #   alignment is captured in delta heading
                desired_velocity = self.desired_velocity(sum_vec, delta_heading)
                # Calculate delta velocity
                delta_velocity = desired_velocity - follower.velocity
                # SAVE DELTA HEADING, DELTA VELOCITY FOR FOLLOWER
                delta_velocities[follower.id] = delta_velocity
                delta_headings[follower.id] = delta_heading
        # Check if any leader actions were input
        if leader_desired_delta_headings is not None and leader_desired_velocities is not None:
            # Go through each leader
            for leader, desired_velocity, delta_heading in zip(self.leaders(), leader_desired_velocities,
                                                               leader_desired_delta_headings):
                # Calculate delta velocity
                delta_velocity = desired_velocity - leader.velocity
                # SAVE DELTA HEADING, DELTA VELOCITY FOR LEADER
                delta_velocities[leader.id] = delta_velocity
                delta_headings[leader.id] = delta_heading
        # Calculate ANGULAR VELOCITY and LINEAR ACCELERATION for each boid (leaders and followers)
        angular_velocities, linear_accelerations = self.kinematics(delta_velocities, delta_headings)
        # Apply angular velocity and linear acceleration to each boid
        self.apply_kinematics(angular_velocities, linear_accelerations)
        return
