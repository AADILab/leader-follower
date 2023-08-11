"""Code for testing different methods in the FitnessCalculator.
Beware, tests passing just means that the tests are passing. It does not mean the methods are completely bug-free
"""
import unittest
from copy import deepcopy

import numpy as np

from lib.poi_colony import POIColony
from lib.boids_colony import BoidsColony
from lib.colony_helpers import BoidsColonyState, StateBounds
from lib.fitness_calculator import FitnessCalculator, WhichG, WhichD, WhichF, FollowerSwitch, PotentialType, UseDrip

class TestFitnessCalculator(unittest.TestCase):
    # def helperTeamFitness(self, poi_colony, boids_colony, position_history, expected_G, expected_G_cs, expected_Ds, which_G, which_D, follower_switch):
    #     """ This method takes in information for a configuration and trajectories and checks if they result in the expected Gs and Ds for a particular G and D combination
    #     """
    #     # Make sure we don't modify any input objects
    #     poi_colony = deepcopy(poi_colony)
    #     position_history = deepcopy(position_history)

    #     potential_arr = []
    #     # Setup the FitnessCalculator
    #     fitness_calculator = FitnessCalculator(poi_colony=poi_colony, boids_colony=boids_colony, which_G=which_G, which_D=which_D, which_F="FCouple", follower_switch=follower_switch)
    #     # distances = fitness_calculator.calculateDistances(poi_colony.pois[0], position_history[3])
    #     # # Distance should be 1.0 at nearest point in trajectory
    #     # self.assertTrue(np.isclose(distances[0], 1.0))
    #     # Calculate continuous G
    #     G = fitness_calculator.calculateG(position_history=position_history)
    #     print(G)
    #     #self.assertTrue(np.isclose(G, expected_G))

    #     print(fitness_calculator.calculateFs(position_history, potential_arr))
    #     boids_colony.updateLeaderInfluence()
    #     print(fitness_calculator.calculateFs(position_history, potential_arr))
    #     print(fitness_calculator.calculateFs(position_history, potential_arr))

    #     # for boid_id in range(position_history.shape[1]):
    #     #     # Calculate counterfactual with the agent removed
    #     #     boid_id = 0
    #     #     G_c = fitness_calculator.calculateCounterfactualG(ids_to_remove=[boid_id], position_history=position_history)
    #     #     self.assertTrue(np.isclose(G_c, expected_G_cs[boid_id]))
    #     #     # Calculate continuous D
    #     #     D = G - G_c
    #     #     # And then D should be 1.0
    #     #     self.assertTrue(np.isclose(D, expected_Ds[boid_id]))

    # def test_MinContinuous_G_D(self):
    #     # Setup pois
    #     poi_colony = POIColony(
    #         positions=np.array([
    #             [2,2]
    #         ]),
    #         # observation radius should not matter for continuous fitness
    #         observation_radius=None,
    #         coupling=1
    #     )
    #     # Setup position history for an agent
    #     position_history = np.array([
    #         [[3,5]],
    #         [[3,4]],
    #         [[3,3]],
    #         [[3,2]],
    #         [[3,1]]
    #     ])

    #     # G should be 1.0 because the fitness is the inverse nearest distance to the poi, and that is 1/1.0
    #     # If we counterfacutally remove the only agent, then the counterfactual score should be 0.0
    #     # And then D should be 1.0
    #     self.helperTeamFitness(poi_colony, None, position_history, 1.0, [0.0], [1.0], WhichG.MinContinuous, WhichD.D, FollowerSwitch.UseLeadersAndFollowers)

    #     # Now we setup the problem again, but with two agents, one traveling on each side of the poi
    #     position_history_2 = np.array([
    #         [[3,5], [1,5]],
    #         [[3,4], [1,4]],
    #         [[3,3], [1,3]],
    #         [[3,2], [1,2]],
    #         [[3,1], [1,1]]
    #     ])

    #     # G should be 1.0 based on proximity of agents to the poi
    #     # If we counterfactually remove one, then the score should be 1.0, regardless of which we remove
    #     # So that makes D for each agent 0.0
    #     self.helperTeamFitness(poi_colony, None, position_history_2, 1.0, [1.0,1.0], [0.0,0.0], WhichG.MinContinuous, WhichD.D, FollowerSwitch.UseLeadersAndFollowers)
    
    # def test_MinContinuous_G_D_Dfollow(self):
    #     # Setup pois
    #     poi_colony = POIColony(
    #         positions=np.array([
    #             [0,2]
    #         ]),
    #         observation_radius=None,
    #         coupling=2
    #     )
    #     # Setup position history
    #     # Leaders on the left, followers on the right
    #     # Just one timestep for this test
    #     position_history=np.array([
    #         [[1,2], [0,0], [1,2], [0,0]]
    #     ])
    #     # [leader, leader, follower, follower]

    #     # Setup minimal boids colony
    #     boids_colony = BoidsColony(
    #         init_state=BoidsColonyState(
    #             positions=position_history[0],
    #             headings=np.array([0,0,0,0]),
    #             velocities=np.array([0,0,0,0]),
    #             is_leader=np.array([True, True, False, False])
    #         ),
    #         bounds=StateBounds(
    #             map_dimensions=np.array([100,100]),
    #             min_velocity=0,
    #             max_velocity=0,
    #             max_acceleration=0,
    #             max_angular_velocity=0,
    #             num_leaders=2,
    #             num_followers=2
    #         ),
    #         radius_repulsion=0,
    #         radius_orientation=0,
    #         radius_attraction=1, # For leader influence for followers
    #         repulsion_multiplier=0,
    #         orientation_multiplier=0,
    #         attraction_multiplier=0,
    #         wall_avoidance_multiplier=0,
    #         dt=0
    #     )

    #     # Update the leader influence for t=0
    #     boids_colony.updateLeaderInfluence()

    #     # Check that the leader influences were updated properly
    #     expected_influences_t0 = [
    #         [1,0],    # Leader influences on the first follower
    #         [0,1]  # Leader influences on the second follower
    #     ]
    #     actual_influences_t0 = [follower.leader_influence for follower in boids_colony.getFollowers()]
    #     self.assertEqual(len(actual_influences_t0), len(expected_influences_t0))
    #     for e_influences, a_influences in zip(expected_influences_t0, actual_influences_t0):
    #         self.assertEqual(len(e_influences), len(a_influences))
    #         for e, a in zip(e_influences, a_influences):
    #             self.assertEqual(e,a)
        
    #     # Setup FitnessCalculator with G and D (regular D, not D_follow)
    #     fitness_calculator = FitnessCalculator(poi_colony=poi_colony, boids_colony=boids_colony, which_G=WhichG.MinContinuous, which_D=WhichD.D, follower_switch=FollowerSwitch.UseLeadersAndFollowers)
    #     # G=1.0
    #     # If we calculate Ds for the leaders, then Ds=[0.33333, 0.0]
    #     G = fitness_calculator.calculateG(position_history)
    #     expected_G = 1.0
    #     self.assertTrue(np.isclose(G, expected_G))
    #     Ds = fitness_calculator.calculateDs(G, position_history)
    #     expected_Ds = [1.0/3.0, 0.0]
    #     self.assertEqual(len(Ds), len(expected_Ds))
    #     for D, expected_D in zip(Ds, expected_Ds):
    #         self.assertTrue(np.isclose(D, expected_D))

    #     # Setup FitnessCalculator with G and D_follow
    #     fitness_calculator_follow = FitnessCalculator(poi_colony=poi_colony, boids_colony=boids_colony, which_G=WhichG.MinContinuous, which_D=WhichD.DFollow, follower_switch=FollowerSwitch.UseLeadersAndFollowers)
    #     # G=1.0
    #     # If we calculate D follow for leaders, then Ds=[0.5, 1.0]
    #     G = fitness_calculator_follow.calculateG(position_history)
    #     expected_G = 1.0
    #     self.assertTrue(np.isclose(G, expected_G))
    #     expected_Ds = [0.5, 0.0]
    #     Ds = fitness_calculator_follow.calculateDs(G, position_history)
    #     self.assertEqual(len(Ds), len(expected_Ds))
    #     for D, expected_D in zip(Ds, expected_Ds):
    #         self.assertTrue(np.isclose(D,expected_D))

    # def test_MinContinous_Dfollow(self):
    #     """This is based on a particular failure where the counterfactual evaluation was somehow greater than the actual evaluation"""
    #     ids_to_remove=[0,2,3]


    # def test_MinDiscrete_G_D(self):
    #     # Setup pois
    #     poi_colony = POIColony(
    #         positions=np.array([
    #             [0,0],
    #             [10,0],
    #             [0,10],
    #             [10,10]
    #         ]),
    #         observation_radius=1,
    #         coupling=3
    #     )
    #     # Setup position histories so half of the pois are observed
    #     # 3 agents are at the first poi, then at the second poi
    #     # The 3 other agents stay far away from the pois and do not observe any
    #     position_history = np.array([
    #         [[0,0],  [0,0],  [0,0],  [100,100], [100,100], [100,100]],
    #         [[10,0], [10,0], [10,0], [100,100], [100,100], [100,100]]
    #     ])
    #     # Now test G and D for this evaluation function
    #     self.helperTeamFitness(
    #         poi_colony=poi_colony,
    #         boids_colony=None,
    #         position_history=position_history,
    #         expected_G=0.5,
    #         expected_G_cs=[0,0,0,0.5,0.5,0.5],
    #         expected_Ds=[0.5,0.5,0.5,0,0,0],
    #         which_G=WhichG.MinDiscrete,
    #         which_D=WhichD.D, # With the existing helper function, this parameter isn't actually used
    #         follower_switch=FollowerSwitch.UseLeadersAndFollowers
    #     )
    
    def test_MinDiscrete_G_FCouple(self):
        # Setup the problem with followers to see if D_follow is calculated properly
        poi_colony = POIColony(
            positions=np.array([
                [0,0],
                [6, 10]
            ]),
            observation_radius=1,
            coupling=1
        )
        

        position_history = np.array([
            [[70,0],  [30,70], [70,0],  [70,0]],
            [[0,0], [6,10], [0,0], [6,10]],
        ])

        potential_vals = []
        # [leader, leader, follower, follower]
        # Setup minimal boids colony
        boids_colony = BoidsColony(
            init_state=BoidsColonyState(
                positions=position_history[0],
                headings=np.array([0,0,0,0]),
                velocities=np.array([0,0,0,0]),
                is_leader=np.array([True, True, False, False])
            ),
            bounds=StateBounds(
                map_dimensions=np.array([100,100]),
                min_velocity=0,
                max_velocity=0,
                max_acceleration=0,
                max_angular_velocity=0,
                num_leaders=2,
                num_followers=2
            ),
            radius_repulsion=0,
            radius_orientation=0,
            radius_attraction=1,
            repulsion_multiplier=0,
            orientation_multiplier=0,
            attraction_multiplier=0,
            wall_avoidance_multiplier=0,
            dt=0,
        )
        fitness_calculator = FitnessCalculator(
            poi_colony=poi_colony, 
            boids_colony=boids_colony, 
            which_G=WhichG.ContinuousObsRadLastStep, 
            which_D=WhichD.DFollow, 
            which_F=WhichF.FCouple,
            follower_switch=FollowerSwitch.UseFollowersOnly,
            potential_type=PotentialType.Global,
            use_drip=UseDrip.No
        )
        subtracted_potentials = []
        

        # Update the leader influence for t=0
        boids_colony.state.positions = position_history[0]
        boids_colony.updateLeaderInfluence()
        print("F is ")
        f_val = fitness_calculator.calculateFs(position_history, potential_vals)
        subtracted_potentials.append(f_val[1])
        print(potential_vals)
        # Check that leader influences were updated properly
        expected_influences_t0 = 0 
        actual_influences_t0 = boids_colony.num_followers_influenced

        boids_colony.state.positions = position_history[1]
        boids_colony.updateLeaderInfluence()
        print("t = 0 G is ")
        g_val = fitness_calculator.calculateG(position_history)
        print(g_val)
        print("D is ")
        Ds = fitness_calculator.calculateDs(g_val, position_history)
        print(Ds)

        print("F is ")
        f_val = fitness_calculator.calculateFs(position_history, potential_vals)
        subtracted_potentials.append(f_val[1])
        print(potential_vals)

        total_F = np.sum(np.array(subtracted_potentials), axis=0)
        Ds = (np.array(Ds) + total_F).tolist()

        print("updated D + F " + str(Ds))

        # boids_colony.state.positions = position_history[1]
        # boids_colony.updateLeaderInfluence()
        # g_val = fitness_calculator.calculateG(position_history)
        # print("t = 1 G is ")
        # print(fitness_calculator.calculateG(position_history))

        # print("D is ")
        # print(fitness_calculator.calculateDs(g_val, position_history, potential_vals))
        # print("F is ")
        # fitness_calculator.calculateFs(position_history, potential_vals)
        # print(potential_vals)





        # print("t = 1")
        # # Update the positions to t=1
        # boids_colony.state.positions = position_history[1]


        # # # Update the leader influence for t=1
        # boids_colony.updateLeaderInfluence()
        # g_val = fitness_calculator.calculateG(position_history)
        # print("t = 1 G is ")
        # print(fitness_calculator.calculateG(position_history))

        # print("t = 1 D + F is ")
        # print(fitness_calculator.calculateDs(g_val, position_history, potential_vals))
        # print("t = 1 potential vals: ")
        # print(potential_vals)

        # boids_colony.updateLeaderInfluence()
        # g_val = fitness_calculator.calculateG(position_history)
        # print("t = 1, G is ")
        # print(fitness_calculator.calculateG(position_history))

        # print("t = 1, D + F is ")
        # print(fitness_calculator.calculateDs(g_val, position_history, potential_vals))


        # print(potential_vals)
        # # Check that leader influences were updated properly
        # expected_influences_t0 = 0 
        # actual_influences_t0 = boids_colony.num_followers_influenced

        # print(fitness_calculator.calculateFs(position_history, potential_vals))
        # print(potential_vals)
        # # # Check that the leader influences were updated properly for t1
        # # expected_influences_t1 = 2
        # # actual_influences_t1 = boids_colony.num_followers_influenced
        # # self.assertEqual(expected_influences_t1, actual_influences_t1)

        # print("t = 2")
        # boids_colony.state.positions = position_history[2]
        # # Update the leader influence for t=1
        # boids_colony.updateLeaderInfluence()
        # print("G is ")
        # print(fitness_calculator.calculateG(position_history))

        # print(fitness_calculator.calculateFs(position_history, potential_vals))
        # print(potential_vals)

        
        # # Check that the leader influences were updated properly for t1
        # expected_influences_t2 = 2
        # actual_influences_t2 = boids_colony.num_followers_influenced
        # self.assertEqual(expected_influences_t2, actual_influences_t2)

        #over here

        #self.helperTeamFitness(poi_colony, boids_colony, position_history, 1.0, [0.0], [1.0], WhichG.ContinuousObsRadLastStep, WhichD.G, FollowerSwitch.UseLeadersAndFollowers)
        

    # def test_MinDiscrete_G_D_Dfollow(self):
    #     # Setup the problem with followers to see if D_follow is calculated properly
    #     poi_colony = POIColony(
    #         positions=np.array([
    #             [0,0],
    #             [10,0],
    #             [0,10],
    #             [10,10]
    #         ]),
    #         observation_radius=1,
    #         coupling=2
    #     )
    #     # Setup position history
    #     # Remember that leaders are the left side, and followers are on the right side
    #     position_history = np.array([
    #         [[0,0],  [0,10],  [0,0],  [0,10]],
    #         [[10,0], [10,10], [10,0], [10,10]]
    #     ])
    #     # [leader, leader, follower, follower]

    #     # Setup minimal boids colony
    #     boids_colony = BoidsColony(
    #         init_state=BoidsColonyState(
    #             positions=position_history[0],
    #             headings=np.array([0,0,0,0]),
    #             velocities=np.array([0,0,0,0]),
    #             is_leader=np.array([True, True, False, False])
    #         ),
    #         bounds=StateBounds(
    #             map_dimensions=np.array([100,100]),
    #             min_velocity=0,
    #             max_velocity=0,
    #             max_acceleration=0,
    #             max_angular_velocity=0,
    #             num_leaders=2,
    #             num_followers=2
    #         ),
    #         radius_repulsion=0,
    #         radius_orientation=0,
    #         radius_attraction=1,
    #         repulsion_multiplier=0,
    #         orientation_multiplier=0,
    #         attraction_multiplier=0,
    #         wall_avoidance_multiplier=0,
    #         dt=0
    #     )
    #     # Update the leader influence for t=0
    #     boids_colony.updateLeaderInfluence()

    #     # Check that leader influences were updated properly
    #     expected_influences_t0 = 0 
    #     actual_influences_t0 = boids_colony.num_followers_influenced

    #     self.assertEqual(expected_influences_t0, actual_influences_t0)
        
        # # Update the positions to t=1
        # boids_colony.state.positions = position_history[1]
        # # Update the leader influence for t=1
        # boids_colony.updateLeaderInfluence()

        # # Check that the leader influences were updated properly for t1
        # expected_influences_t1 = [
        #     [2, 0],
        #     [0, 2]
        # ]
        # actual_influences_t1 = [follower.leader_influence for follower in boids_colony.getFollowers()]
        # self.assertEqual(len(actual_influences_t1), len(expected_influences_t1))
        # for e_influences, a_influences in zip(expected_influences_t1, actual_influences_t1):
        #     self.assertEqual(len(e_influences), len(a_influences))
        #     for e, a in zip(e_influences, a_influences):
        #         self.assertEqual(e,a)

        # # Setup the FitnessCalculator with G and D (regular D, not D_follow)
        # fitness_calculator = FitnessCalculator(poi_colony=poi_colony, boids_colony=boids_colony, which_G=WhichG.MinDiscrete, which_D=WhichD.D, follower_switch=FollowerSwitch.UseLeadersAndFollowers)
        # # G=1.0
        # # If we calculate Ds, for the leaders, then Ds=[0.5, 0.5]
        # G = fitness_calculator.calculateG(position_history)
        # expected_G = 1.0
        # self.assertTrue(np.isclose(G,expected_G))
        # Ds = fitness_calculator.calculateDs(G, position_history)
        # expected_Ds = [0.5, 0.5]
        # self.assertEqual(len(Ds), len(expected_Ds))
        # for D, expected_D in zip(Ds, expected_Ds):
        #     self.assertTrue(np.isclose(D, expected_D))

        # # Setup the fitness calculator to see if we can calculate D_follow
        # fitness_calculator_follower = FitnessCalculator(poi_colony=poi_colony, boids_colony=boids_colony, which_G=WhichG.MinDiscrete, which_D=WhichD.D, follower_switch=FollowerSwitch.UseLeadersAndFollowers)
        # # G=1.0
        # # If we're only looking at D_follow for the leaders, then we should see (again) 
        # # Ds = [0.5, 0.5]
        # G = fitness_calculator_follower.calculateG(position_history)
        # expected_G = 1.0
        # self.assertTrue(np.isclose(G, expected_G))
        # Ds = fitness_calculator_follower.calculateDs(G, position_history)
        # expected_Ds = [0.5, 0.5]
        # self.assertEqual(len(Ds), len(expected_Ds))
        # for D, expected_D in zip(Ds, expected_Ds):
        #     self.assertTrue(np.isclose(D, expected_D))
        
    # def test_follower_switch(self):
    #     # Setup Pois
    #     poi_colony = POIColony(
    #         # Place two pois vertically stacked
    #         positions=np.array([
    #             [10, 10],
    #             [10, 13]
    #         ]),
    #         # Observation radius to see them from anywhere
    #         observation_radius=100,
    #         # 2 agents need to observe each poi
    #         coupling=2
    #     )
    #     # Setup position history
    #     # Remember that leaders are the left side, and followers are on the right side
    #     position_history = np.array([
    #         [[10, 9], [10, 14], [10, 11], [10,12]]
    #     ])
    #     # [leader, leader, follower, follower]
    #     # Followers are in between the pois.
    #     # Leaders are on the outside

    #     # Setup minimal boids colony
    #     boids_colony = BoidsColony(
    #         init_state=BoidsColonyState(
    #             positions=position_history[0],
    #             headings=np.array([0,0,0,0]),
    #             velocities=np.array([0,0,0,0]),
    #             is_leader=np.array([True, True, False, False])
    #         ),
    #         bounds=StateBounds(
    #             map_dimensions=np.array([100,100]),
    #             min_velocity=0,
    #             max_velocity=0,
    #             max_acceleration=0,
    #             max_angular_velocity=0,
    #             num_leaders=2,
    #             num_followers=2
    #         ),
    #         radius_repulsion=0,
    #         radius_orientation=0,
    #         radius_attraction=1,
    #         repulsion_multiplier=0,
    #         orientation_multiplier=0,
    #         attraction_multiplier=0,
    #         wall_avoidance_multiplier=0,
    #         dt=0
    #     )

    #     # Setup the FitnessCalculator with G and D
    #     # Set it up first to count leaders and followers
    #     fitness_calculator = FitnessCalculator(
    #         poi_colony=poi_colony, 
    #         boids_colony=boids_colony, 
    #         which_G=WhichG.ContinuousObsRadLastStep, 
    #         which_D=WhichD.D, 
    #         follower_switch=FollowerSwitch.UseLeadersAndFollowers
    #     )

    #     # Calculate G when we count the leaders and followers
    #     G = fitness_calculator.calculateG(position_history)
    #     expected_G = 1.0
    #     self.assertTrue(np.isclose(G,expected_G))

    #     # Set up the fitness calculator to instead just count followers
    #     fitness_calculator.follower_switch=FollowerSwitch.UseFollowersOnly

    #     # Calculate G with just followers
    #     G = fitness_calculator.calculateG(position_history)
    #     expected_G = 2./3.
    #     self.assertTrue(np.isclose(G, expected_G))

    # def test_d_zero(self):
    #     # Setup Pois
    #     poi_colony = POIColony(
    #         # Place two pois vertically stacked
    #         positions=np.array([
    #             [10, 10],
    #             [10, 13]
    #         ]),
    #         # Observation radius to see them from anywhere
    #         observation_radius=100,
    #         # 2 agents need to observe each poi
    #         coupling=2
    #     )
    #     # Setup position history
    #     # Remember that leaders are the left side, and followers are on the right side
    #     position_history = np.array([
    #         [[10, 9], [10, 14], [10, 11], [10,12]]
    #     ])
    #     # [leader, leader, follower, follower]
    #     # Followers are in between the pois.
    #     # Leaders are on the outside

    #     # Setup minimal boids colony
    #     boids_colony = BoidsColony(
    #         init_state=BoidsColonyState(
    #             positions=position_history[0],
    #             headings=np.array([0,0,0,0]),
    #             velocities=np.array([0,0,0,0]),
    #             is_leader=np.array([True, True, False, False])
    #         ),
    #         bounds=StateBounds(
    #             map_dimensions=np.array([100,100]),
    #             min_velocity=0,
    #             max_velocity=0,
    #             max_acceleration=0,
    #             max_angular_velocity=0,
    #             num_leaders=2,
    #             num_followers=2
    #         ),
    #         radius_repulsion=0,
    #         radius_orientation=0,
    #         radius_attraction=1,
    #         repulsion_multiplier=0,
    #         orientation_multiplier=0,
    #         attraction_multiplier=0,
    #         wall_avoidance_multiplier=0,
    #         dt=0
    #     )

    #     # Setup the FitnessCalculator with G and D
    #     # Set it up first to count leaders and followers
    #     fitness_calculator = FitnessCalculator(
    #         poi_colony=poi_colony, 
    #         boids_colony=boids_colony, 
    #         which_G=WhichG.ContinuousObsRadLastStep, 
    #         which_D=WhichD.Zero, 
    #         follower_switch=FollowerSwitch.UseLeadersAndFollowers
    #     )


    #     # Calculate G when we count the leaders and followers
    #     G = fitness_calculator.calculateG(position_history)

    #     # Just make sure Ds are zeros
    #     Ds = fitness_calculator.calculateDs(G, position_history)
    #     self.assertTrue(np.allclose(Ds, np.zeros(2)))

if __name__ == '__main__':
    unittest.main()
