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
        repulsion_boids, orientation_boids, attraction_boids = bm.update_follower_observations()
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
        desired_headings, velocities = bm.update_follower_actions(repulsion_boids, orientation_boids, attraction_boids)
        self.assertTrue(type(desired_headings)==np.ndarray)
        self.assertTrue(type(velocities)==np.ndarray)
        self.assertTrue(desired_headings.shape == (num_followers,1))
        self.assertTrue(velocities.shape == (num_followers,1))

        # Apply actions
        bm.apply_follower_actions(desired_headings, velocities)

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
        ])
        # All boids facing right except for 2nd boid, which is facing up
        headings = np.array([
            [0],
            [0],
            [np.pi/2],
            [0]
        ])
        # All boids start at rest
        velocities = np.array([
            [0],
            [0],
            [0],
            [0]
        ])
        bm = BoidsManager(max_velocity, angular_velocity, \
            radius_repulsion, radius_orientation, radius_attraction, \
                num_followers, num_leaders, map_size,
                positions=positions, headings=headings, velocities=velocities)
        # Step the simulation forward one timestep. Go step by step
        # Update observations
        all_obs_rep_boids_pos, all_obs_orient_boids_head, all_obs_attract_boids_pos = bm.update_follower_observations()
        # Check that Boid 0's observations are as expected
        self.assertTrue(np.all(all_obs_rep_boids_pos[0]==np.asarray([[50, 48]])))
        self.assertTrue(all_obs_orient_boids_head[0]==np.asarray([[np.pi/2]]))
        self.assertTrue(np.all(all_obs_attract_boids_pos[0]==np.asarray([[60,50]])))
        # Update desired actions
        desired_headings, desired_velocities = bm.update_follower_actions(all_obs_rep_boids_pos, all_obs_orient_boids_head, all_obs_attract_boids_pos)
        print("desired_headings:\n",desired_headings)
        print("velocities:\n",desired_velocities)

        pass

    def test_boid_behavior_many(self):
        """Test if a boid behaves as expected when acted upon by
        a three boids repulsing
        a three boids orienting
        a three boids attracting
        """
        pass

if __name__ == '__main__':
    unittest.main()
