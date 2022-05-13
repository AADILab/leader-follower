import unittest
import numpy as np
import pygame
from time import time

from renderer import Renderer

class TestRenderer(unittest.TestCase):
    def continuouslyRender(self, r, positions, headings):
        delay_time = 0.1
        last_time = None

        shutdown = False
        while not shutdown:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    shutdown = True
            current_time = time()
            if last_time is None or current_time - last_time > delay_time:
                last_time = current_time
                r.renderFrame(positions, headings)

    def testBoidPlacement(self):
        positions = np.array([
            [5,10],
            [15,10],
            [10,15],
            [10,5]
        ])
        headings = np.array([
            [np.pi],
            [0],
            [np.pi/2],
            [3*np.pi/2]
        ])
        num_leaders = 1
        num_followers = 3
        map_size = np.array([100,50])
        pixels_per_unit = 20
        r = Renderer(num_leaders, num_followers, map_size, pixels_per_unit)
        self.continuouslyRender(r, positions, headings)

if __name__ == '__main__':
    unittest.main()
