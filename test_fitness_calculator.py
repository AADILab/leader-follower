import unittest
from copy import deepcopy

import numpy as np

from lib.poi_colony import POIColony
from lib.boids_colony import BoidsColony
from lib.fitness_calculator import FitnessCalculator

class TestContinousFitness(unittest.TestCase):
    def helperTeamFitness(self, poi_colony, position_history, expected_G, expected_G_cs, expected_Ds):
        # Make sure we don't modify any input objects
        poi_colony = deepcopy(poi_colony)
        position_history = deepcopy(position_history)
        # Setup the FitnessCalculator
        fitness_calculator = FitnessCalculator(poi_colony=poi_colony, boids_colony=None)
        # distances = fitness_calculator.calculateDistances(poi_colony.pois[0], position_history[3])
        # # Distance should be 1.0 at nearest point in trajectory
        # self.assertTrue(np.isclose(distances[0], 1.0))
        # Calculate continuous G
        G = fitness_calculator.calculateContinuousTeamFitness(poi_colony=None, position_history=position_history)
        
        self.assertTrue(np.isclose(G, expected_G))

        for boid_id in range(position_history.shape[1]):
            # Calculate counterfactual with the agent removed
            G_c = fitness_calculator.calculateCounterfactualTeamFitness(boid_id=0, position_history=position_history)
            self.assertTrue(np.isclose(G_c, expected_G_cs[boid_id]))
            # Calculate continuous D
            D = G - G_c
            # And then D should be 1.0
            self.assertTrue(np.isclose(D, expected_Ds[boid_id]))

    def test_teamFitness(self):
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
        position_history = np.array([
            [[3,5]],
            [[3,4]],
            [[3,3]],
            [[3,2]],
            [[3,1]]
        ])

        # G should be 1.0 because the fitness is the inverse nearest distance to the poi, and that is 1/1.0
        # If we counterfacutally remove the only agent, then the counterfactual score should be 0.0
        # And then D should be 1.0
        self.helperTeamFitness(poi_colony, position_history, 1.0, [0.0], [1.0])

        # Now we setup the problem again, but with two agents, one traveling on each side of the poi
        position_history_2 = np.array([
            [[3,5], [1,5]],
            [[3,4], [1,4]],
            [[3,3], [1,3]],
            [[3,2], [1,2]],
            [[3,1], [1,1]]
        ])
        print(position_history.shape)

        # G should be 1.0 based on proximity of agents to the poi
        # If we counterfactually remove one, then the score should be 1.0, regardless of which we remove
        # So that makes D for each agent 0.0
        self.helperTeamFitness(poi_colony, position_history_2, 1.0, [1.0,1.0], [0.0,0.0])

if __name__ == '__main__':
    unittest.main()