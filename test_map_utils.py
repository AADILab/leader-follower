from tkinter import FALSE
from turtle import pos
import unittest
import numpy as np

from map_utils import Map

class TestMap(unittest.TestCase):
    def test_map_object(self):
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
        # self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()