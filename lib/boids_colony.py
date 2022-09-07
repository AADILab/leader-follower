from typing import List, Tuple

import numpy as np

def calculateDistance(positions_a, positions_b):
    return np.linalg.norm(positions_a-positions_b, axis=1)

def calculateDeltaHeading(current_heading, desired_heading):
    """ Calculate delta headings such that delta is the shortest path from
    current heading to the desired heading.
    """
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
    return delta_heading

class Boid():
    def __init__(self, positions, headings, velocities, is_leader, id) -> None:
        self._positions = positions
        self._headings = headings
        self._velocities = velocities
        self._is_leader = is_leader
        self.id = id

    @property
    def position(self):
        return self._positions[self.id]

    @property
    def heading(self):
        return self._headings[self.id]

    @property
    def velocity(self):
        return self._velocities[self.id]

    def isLeader(self):
        return self._is_leader[self.id]

BoidArray = np.ndarray[Boid, np.dtype[Boid]]

class BoidsColony():
    def __init__(self,
        leader_positions: List[List[float]], follower_positions: List[List[float]],
        leader_headings: List[float], follower_headings: List[float],
        leader_velocities: List[float], follower_velocities: List[float],
        radius_repulsion: float, radius_orientation: float, radius_attraction: float,
        repulsion_mulitplier: float,
        map_dimensions: List[float],
        min_velocity: float, max_velocity: float,
        max_acceleration: float,
        max_angular_velocity: float,
        dt: float
        ) -> None:

        self.num_leaders = len(leader_positions)
        self.num_followers = len(follower_positions)
        self.num_total = self.num_leaders+self.num_followers

        self.is_leader = np.array([True for _ in range(self.num_leaders)]+[False for _ in range(self.num_followers)])

        self.positions = np.array(leader_positions+follower_positions, dtype=float)
        self.headings = np.array(leader_headings+follower_headings, dtype=float)
        self.velocities = np.array(leader_velocities+follower_velocities, dtype=float)

        self.boids = np.array([Boid(self.positions, self.headings, self.velocities, self.is_leader, id) for id in range(self.num_total)])

        self.radius_repulsion = radius_repulsion
        self.radius_orientation = radius_orientation
        self.radius_attraction = radius_attraction

        self.repulsion_multiplier = repulsion_mulitplier

        self.map_dimensions = np.array(map_dimensions, dtype=float)

        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity
        self.dt = dt

        for boid in self.boids:
            print(boid.position, boid.heading, boid.velocity)

    def getLeaders(self):
        return self.boids[:self.num_leaders]

    def getFollowers(self):
        return self.boids[self.num_leaders:]

    def getObservableBoids(self, boid: Boid, return_distances=False):
        """Get all boids observable by this boid"""
        distances = calculateDistance(self.positions, boid.position)
        observable_bool = distances <= self.radius_attraction
        observable_bool[boid.id] = False
        if return_distances:
            return self.boids[observable_bool], distances[observable_bool]
        return self.boids[observable_bool]

    def splitRepOriAtt(self, observable_boids, distances):
        """Split observable boids into repulsion, orientation, and attraction boids"""
        repulsion_bool = distances <= self.radius_repulsion
        orientation_bool = np.logical_and(distances > self.radius_repulsion, distances <= self.radius_orientation)
        attraction_bool = np.logical_and(distances > self.radius_orientation, distances <= self.radius_attraction)

        repulsion_boids = observable_boids[repulsion_bool]
        orientation_boids = observable_boids[orientation_bool]
        attraction_boids = observable_boids[attraction_bool]

        return repulsion_boids, orientation_boids, attraction_boids

    def repulsionVec(self, boid: Boid, repulsion_boids: List[Boid]):
        # Repulsion vector is average vector from repulsion boids to current boid, normalized by radius of repulsion
        repulsion_positions = np.array([boid.position for boid in repulsion_boids])
        if np.shape(repulsion_positions)[0] != 0:
            return (boid.position - np.average(repulsion_positions, axis=0))/self.radius_repulsion * self.repulsion_multiplier
        else:
            return np.array([0,0])

    def orientationVec(self, orientation_boids: List[Boid]):
        # Orientation vector is sum of vectors derived from orientations of orientation boids
        orientation_headings = np.array([boid.heading for boid in orientation_boids])
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

    def attractionVec(self, boid: Boid, attraction_boids: List[Boid]):
        # Attraction vector is average vector from current boid to attraction boids, normalized by radius of attraction
        attraction_positions = np.array([boid.position for boid in attraction_boids])
        if np.shape(attraction_positions)[0] != 0:
            return (np.average(attraction_positions, axis=0) - boid.position)/self.radius_attraction
        else:
            return np.array([0,0])

    def nearObservableWall(self, boid: Boid):
        if np.any(boid.position <= self.radius_repulsion):
            return True
        elif np.any(boid.position > self.map_dimensions - self.radius_repulsion):
            return True
        else:
            return False

    def nearObservableBoids(self, boid: Boid):
        return len(self.getObservableBoids(boid)) == 0

    def wallAvoidanceVec(self, boid: Boid):
        wall_vec = np.array([0,0], dtype=float)
        # Left wall
        if boid.position[0] <= self.radius_repulsion:
            wall_vec[0] = self.radius_repulsion - boid.position[0]
        # Right wall
        elif boid.position[0] >= self.map_dimensions[0] - self.radius_repulsion:
            wall_vec[0] = self.map_dimensions[0] - self.radius_repulsion - boid.position[0]
        # Bottom wall
        if boid.position[1] <= self.radius_repulsion:
            wall_vec[1] = self.radius_repulsion - boid.position[1]
        # Top wall
        elif boid.position[1] >= self.map_dimensions[1] - self.radius_repulsion:
            wall_vec[1] = self.map_dimensions[1] - self.radius_repulsion - boid.position[1]
        return wall_vec

    def calculateDesiredVelocity(self, sum_vector, delta_heading):
        # Heading is well aligned
        if np.abs(delta_heading) < np.pi/2:
            # Match desired velocity to magnitude of sum vector
            return np.linalg.norm(sum_vector)
        # Heading is not well aligned
        else:
            # Slow down
            return self.min_velocity

    def calculateKinematics(self, delta_velocities, delta_headings):
        """Turn deltas for heading and velocity into angular velocities
        and linear acclerations. Bound kinematics according to specified
        boundaries. Ex: max acceleration, max angular velocity
        """
        angular_velocities = delta_headings/self.dt
        angular_velocities[angular_velocities > self.max_angular_velocity] = self.max_angular_velocity
        angular_velocities[angular_velocities < -self.max_angular_velocity] = -self.max_angular_velocity
        linear_accelerations = delta_velocities/self.dt
        linear_accelerations[linear_accelerations > self.max_acceleration] = self.max_acceleration
        linear_accelerations[linear_accelerations < -self.max_acceleration] = -self.max_acceleration
        return angular_velocities, linear_accelerations

    def applyKinematics(self, angular_velocities, linear_accelerations):
        """Update all positions, velocities, and headings with the input kinematics using Euler integration.
        Bound kinematics according to specified boundaries. Ex: max_velocity
        """
        # Update headings
        self.headings += angular_velocities*self.dt
        # Apply circular cutoff
        self.headings %= (2*np.pi)
        # Update velocities
        self.velocities += linear_accelerations*self.dt
        self.velocities[self.velocities > self.max_velocity] = self.max_velocity
        self.velocities[self.velocities < self.min_velocity] = self.min_velocity
        # Update positions
        self.positions[:,0] += self.velocities * np.cos(self.headings) * self.dt
        self.positions[:,1] += self.velocities * np.sin(self.headings) * self.dt
        # Bound positions
        # Apply left bound
        self.positions[:,0][self.positions[:,0]<0] = 0
        # Apply right bound
        self.positions[:,0][self.positions[:,0]>self.map_dimensions[0]] = self.map_dimensions[0]
        # Apply lower bound
        self.positions[:,1][self.positions[:,1]<0] = 0
        # Apply upper bound
        self.positions[:,1][self.positions[:,1]>self.map_dimensions[1]] = self.map_dimensions[1]

    def step(self, leader_desired_velocities, leader_desired_headings):
        # Initialize desired velocities array
        # Initialize desired headings array
        delta_velocities = np.zeros(self.num_total)
        delta_headings = np.zeros(self.num_total)

        vels = []

        # Go through each follower
        for follower in self.getFollowers():
            # Get all boids within observation radius
            observable_boids, distances = self.getObservableBoids(follower, return_distances=True)
            # Determine if boid is near wall
            near_wall = self.nearObservableWall(follower)
            # If no boids are observed, and this boid is not near a wall
            if len(observable_boids) == 0 and not near_wall:
                # Calculate its delta velocity as 0.0 and delta heading as 0.0
                delta_velocities[follower.id] = 0.0
                delta_headings[follower.id] = 0.0
            else:
                # Seperate observable boids into repulsion, orientation, and attraction boids
                rep_boids, ori_boids, att_boids = self.splitRepOriAtt(observable_boids, distances)
                # Calculate repulsion vector
                repulsion_vec = self.repulsionVec(follower, rep_boids)
                # Calculate orientation vector
                orientation_vec = self.orientationVec(ori_boids)
                # Calculate attraction vector
                attraction_vec = self.attractionVec(follower, att_boids)
                # Calculate wall avoidance vector
                wall_avoid_vec = self.wallAvoidanceVec(follower)
                # Sum vectors together to get x,y vector representing desired trajectory
                sum_vec = repulsion_vec + orientation_vec + attraction_vec + wall_avoid_vec
                print(sum_vec)
                # X, Y VECTOR CALCULATED
                # Calculate a desired heading based on x,y vector
                desired_heading = np.arctan2(sum_vec[1], sum_vec[0])
                # Calculate delta heading required to get desired heading
                delta_heading = calculateDeltaHeading(follower.heading, desired_heading)
                # Calculate desired velocity based on alignment between current heading and desired heading (alignment is captured in delta heading)
                desired_velocity = self.calculateDesiredVelocity(sum_vec, delta_heading)
                vels.append(desired_velocity)
                # Calculate delta velocity
                delta_velocity = desired_velocity - follower.velocity
                # SAVE DELTA HEADING, DELTA VELOCITY FOR FOLLOWER
                delta_velocities[follower.id] = delta_velocity
                delta_headings[follower.id] = delta_heading
            # print(delta_velocity, delta_heading)

        print("vels:")
        print(vels)
        print("Deltas:")
        for delta_heading, delta_vel in zip(delta_headings, delta_velocities):
            print(delta_heading, delta_vel)
        # Go through each leader
        for leader, desired_velocity, desired_heading in zip(self.getLeaders(), leader_desired_velocities, leader_desired_headings):
            # Calculate delta heading, delta velocity
            delta_heading = calculateDeltaHeading(leader.heading, desired_heading)
            delta_velocity = desired_velocity - leader.velocity
            # SAVE DELTA HEADING, DELTA VELOCITY FOR LEADER
            delta_velocities[leader.id] = delta_velocity
            delta_heading[leader.id] = delta_heading
        # Calculate ANGULAR VELOCITY and LINEAR ACCELERATION for each boid (leaders and followers)
        angular_velocities, linear_accelerations = self.calculateKinematics(delta_velocities, delta_headings)
        print("Kinematics:")
        for boid in self.boids:
            print(angular_velocities[boid.id], linear_accelerations[boid.id])
        # Apply angular velocity and linear acceleration to each boid
        self.applyKinematics(angular_velocities, linear_accelerations)

if __name__ == "__main__":
    a=np.array([Boid(None, None, None, None, None)])
