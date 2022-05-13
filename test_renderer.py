import unittest
import numpy as np
import pygame
from time import time

from renderer import Renderer

class TestRenderer(unittest.TestCase):
    def continuouslyRender(self, r, positions, headings):
        delay_time = 0.1
        last_time = -delay_time

        shutdown = False
        while not shutdown:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    shutdown = True
                current_time = time()
                if current_time - last_time > delay_time:
                    last_time = current_time
                    r.renderFrame(positions, headings)

    def testBoidPlacement(self):
        positions = np.array([
            [90,10],
            [10,10]
        ])
        headings = np.array([
            [0],
            [np.pi]
        ])
        num_leaders = 1
        num_followers = 1
        map_size = np.array([200,100])
        pixels_per_unit = 10
        r = Renderer(num_leaders, num_followers, map_size, pixels_per_unit)
        self.continuouslyRender(r, positions, headings)

if __name__ == '__main__':
    unittest.main()
