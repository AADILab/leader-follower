import unittest

import numpy as np

from lib.poi_colony import POIColony

class TestContinousFitness(unittest.TestCase):
    def teamFitness(self):
        # Setup pois
        poi_colony = POIColony(
            positions=np.array([
                [2,2]
            ]),
            # observation radius should not matter for continuous fitness
            observation_radius=None,
            coupling=1
        )
        # Setup position history for an agent
        

        # Setup the FitnessCalculator


        