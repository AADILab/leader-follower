import unittest
import numpy as np

from boids_manager import BoidsManager

class TestBoidsManager(unittest.TestCase):
    @staticmethod
    def is_list_of_arrays(input_list):
        list_of_arr = True
        for entry in input_list:
            if type(entry) != np.ndarray:
                list_of_arr = False
        return list_of_arr

    def test_basics(self):
        """ Test basic functionality of a full case
        """
        max_velocity = 1 # 1 unit/step
        angular_velocity = np.pi/32 # rad/step
        radius_repulsion = 2 # units
        radius_orientation = 5 # units
        radius_attraction = 10 # units
        num_followers = 45
        num_leaders = 5
        map_size = np.array([100,100])
        bm = BoidsManager(max_velocity, angular_velocity, \
            radius_repulsion, radius_orientation, radius_attraction, \
                num_followers, num_leaders, map_size)

        # Check that positions and headings are the correct size
        self.assertTrue(bm.positions.shape == (num_leaders+num_followers, 2))
        self.assertTrue(bm.headings.shape == (num_leaders+num_followers, 1))

        # Update the observations
        repulsion_boids, orientation_boids, attraction_boids = bm.get_follower_observations()

        # Each output is a list where the index corresponds to a particular boid
        self.assertTrue(type(repulsion_boids)==list)
        self.assertTrue(type(orientation_boids)==list)
        self.assertTrue(type(attraction_boids)==list)

        # Each element of each list is an array of boids that repulse, orient, or attract
        # the boid with the matching index
        self.assertTrue(self.is_list_of_arrays(repulsion_boids))
        self.assertTrue(self.is_list_of_arrays(orientation_boids))
        self.assertTrue(self.is_list_of_arrays(attraction_boids))

        # Update the actions
        desired_headings, velocities = bm.calculate_follower_desired_actions(repulsion_boids, orientation_boids, attraction_boids)
        self.assertTrue(type(desired_headings)==np.ndarray)
        self.assertTrue(type(velocities)==np.ndarray)
        self.assertTrue(desired_headings.shape == (num_followers,1))
        self.assertTrue(velocities.shape == (num_followers,1))

        # Apply actions
        bm.update_follower_states(desired_headings, velocities)

    def test_boid_behavior(self):
        """Test if a boid behaves as expected when acted upon by
        a single boid repulsing
        a single boid orienting
        a single boid attracting
        """
        max_velocity = 1 # 1 unit/step
        angular_velocity = np.pi/32 # rad/step
        radius_repulsion = 2 # units
        radius_orientation = 5 # units
        radius_attraction = 10 # units
        num_followers = 4
        num_leaders = 0
        map_size = np.array([100,100])
        positions = np.array([
            [50,50],
            [50,48],
            [55,50],
            [60,50]
        ], dtype=np.float64)
        # All boids facing right except for 2nd boid, which is facing up
        headings = np.array([
            [0],
            [0],
            [np.pi/2],
            [0]
        ])
        bm = BoidsManager(max_velocity, angular_velocity, \
            radius_repulsion, radius_orientation, radius_attraction, \
                num_followers, num_leaders, map_size,
                positions=positions, headings=headings)

        # Step the simulation forward one timestep. Go step by step
        # Update observations
        all_obs_rep_boids_pos, all_obs_orient_boids_head, all_obs_attract_boids_pos = bm.get_follower_observations()

        # Check that Boid 0's observations are as expected
        self.assertTrue(np.all(all_obs_rep_boids_pos[0]==np.array([[50, 48]])))
        self.assertTrue(all_obs_orient_boids_head[0]==np.array([[np.pi/2]]))
        self.assertTrue(np.all(all_obs_attract_boids_pos[0]==np.array([[60,50]])))

        # Update desired actions
        all_desired_headings, all_desired_velocities, all_sum_vectors, \
            all_repulsion_vectors, all_orientation_vectors, all_attraction_vectors \
                = bm.calculate_follower_desired_actions(all_obs_rep_boids_pos, all_obs_orient_boids_head, all_obs_attract_boids_pos, debug=True)

        # Check that Boid 0's repulsion, orientation, and attraction vectors are as expected
        self.assertTrue(np.all(all_repulsion_vectors[0]==np.array([[0,1]])))
        self.assertTrue(np.allclose(all_orientation_vectors[0],np.array([[0,1]])))
        self.assertTrue(np.all(all_attraction_vectors[0]==np.array([[1,0]])))

        # Check that Boid 0's total vector is as expected
        self.assertTrue(np.all(all_sum_vectors[0]==np.array([[1,2]])))

        # Check that Boid 0's desired velocity and heading are as expected
        expected_desired_heading = np.arctan2(2,1)
        expected_desired_velocity = np.sqrt(1**2 + 2**2)
        self.assertTrue(all_desired_headings[0]==[expected_desired_heading])
        self.assertTrue(all_desired_velocities[0]==[expected_desired_velocity])

        # Update states of all boids with desired headings and velocities
        bm.update_follower_states(all_desired_headings, all_desired_velocities)

        # Check that Boid 0 has moved accordingly
        self.assertTrue(bm.headings[0]==[np.pi/32])
        self.assertTrue(np.all(bm.positions[0]==[50+np.cos(np.pi/32), 50+np.sin(np.pi/32)]))

    def test_ghost_boids(self):
        """Test that counterfactual ghost boids are generated correctly
        and that boids avoid them as expected.
        """
        # Define variables for creating boids managers
        max_velocity = 1 # 1 unit/step
        angular_velocity = np.pi/32 # rad/step
        radius_repulsion = 1 # units
        radius_orientation = 2 # units
        radius_attraction = 3 # units
        num_followers = 1
        num_leaders = 0
        map_size = np.array([3,2])
        positions = np.array([
            [1,1]
        ], dtype=np.float64)
        headings = np.array([
            [0]
        ])
        # Create a boids mananger with a ghost density of 1
        bm_1_density = BoidsManager(max_velocity, angular_velocity, \
            radius_repulsion, radius_orientation, radius_attraction, \
            num_followers, num_leaders, map_size,
            positions=positions, headings=headings, \
            avoid_walls=True, ghost_density=1)
        exp_ghost_positions_1 = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [0, 2],
            [1, 2],
            [2, 2],
            [3, 2],
            [0, 1],
            [3, 1]
        ])
        # Check that ghost boids are correctly generated with ghost density of 1
        self.assertTrue(np.all(bm_1_density.ghost_positions==exp_ghost_positions_1))

        # Create a boids manager with a ghost density of 2
        bm_2_density = BoidsManager(max_velocity, angular_velocity, \
            radius_repulsion, radius_orientation, radius_attraction, \
            num_followers, num_leaders, map_size,
            positions=positions, headings=headings, \
            avoid_walls=True, ghost_density=2)
        exp_ghost_positions_2 = np.array([
            [0. , 0. ],
            [0.5, 0. ],
            [1. , 0. ],
            [1.5, 0. ],
            [2. , 0. ],
            [2.5, 0. ],
            [3. , 0. ],
            [0. , 2. ],
            [0.5, 2. ],
            [1. , 2. ],
            [1.5, 2. ],
            [2. , 2. ],
            [2.5, 2. ],
            [3. , 2. ],
            [0. , 0.5],
            [0. , 1. ],
            [0. , 1.5],
            [3. , 0.5],
            [3. , 1. ],
            [3. , 1.5]
        ])
        # Check that ghost boids are correctly generated with ghost density of 2
        self.assertTrue(np.all(bm_2_density.ghost_positions==exp_ghost_positions_2))

        # Check that the non-ghost boid is repulsed by the ghost boids
        # The boid is at [1,1] with radius of repulsion of 1.
        # The ghost boids at [1,0], [0,1], [1,2] should repulse the non-ghost boid.
        all_obs_rep_boids_pos, all_obs_orient_boids_head, all_obs_attract_boids_pos = bm_2_density.get_follower_observations()
        # Check that ghosts were observed only in repulsion
        self.assertTrue(all_obs_orient_boids_head[0].size == 0)
        self.assertTrue(all_obs_attract_boids_pos[0].size == 0)
        exp_obs_rep_boid_pos = np.array([
            [1,0],
            [0,1],
            [1,2]
        ])
        self.assertTrue(np.all(all_obs_rep_boids_pos[0]==exp_obs_rep_boid_pos))

    def test_boid_behavior_many(self):
        """Test if a boid behaves as expected when acted upon by
        a three boids repulsing
        a three boids orienting
        a three boids attracting
        """
        pass

if __name__ == '__main__':
    unittest.main()
