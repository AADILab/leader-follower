from tkinter import FALSE
from turtle import pos
import unittest
import numpy as np

from map_utils import Map

class TestMap(unittest.TestCase):
    def test_bin_placement(self):
        map_size = np.array([10,20])
        positions = np.array([
            [10,10],
            [10,20],
            [5,5],
            [1,1],
            [0,0]
        ])
        observation_radius = 5
        boid_map = Map(map_size, observation_radius, positions)
        self.assertTrue(boid_map.bins[0,0] == [2,3,4])
        self.assertTrue(boid_map.bins[1,1] == [0])
        self.assertTrue(boid_map.bins[1,3] == [1])

    def test_get_adj_bins(self):
        map_size = np.array([10,10])
        positions = []
        observation_radius = 1
        boid_map = Map(map_size, observation_radius, positions)
        bin_locations = [
            [0,0],
            [9,9],
            [4,5]
        ]
        for bin_location in bin_locations:
            adj_bins = boid_map.get_adj_bins(bin_location)
            # Test origin (bottom left) edge case
            if bin_location == [0,0]:
                exp_bins = [
                    [0,0],
                    [0,1],
                    [1,1],
                    [1,0]
                ]
                self.assertTrue(np.array_equal(adj_bins, exp_bins))
            # Test top right edge case
            elif bin_location == [9,9]:
                exp_bins = [
                    [9,9],
                    [9,8],
                    [8,8],
                    [8,9]
                ]
                self.assertTrue(np.array_equal(adj_bins, exp_bins))
            # Test normal case
            elif bin_location == [4,5]:
                exp_bins = [
                    [4, 5],
                    [4, 6],
                    [5, 6],
                    [5, 5],
                    [5, 4],
                    [4, 4],
                    [3, 4],
                    [3, 5],
                    [3, 6]
                ]
                self.assertTrue(np.array_equal(adj_bins, exp_bins))

if __name__ == '__main__':
    unittest.main()
