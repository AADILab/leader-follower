import unittest

import numpy as np

from lib.np_helpers import invertInds

class TestNPHelpers(unittest.TestCase):
    def test_invertInds(self):
        dimension = 8
        ids_to_remove = [0,3,5]
        ids_to_keep = invertInds(dimension, ids_to_remove)
        expected_ids_to_keep = [1,2,4,6,7]
        self.assertListEqual(ids_to_keep, expected_ids_to_keep)

    def test_NPSlicing(self):
        # Make a test position history with fake values, 
        # but with reasonable dimensions
        # 100 timesteps, 8 agents, 2 for xy
        position_history = np.ones((100,8,2))
        ids_to_remove = [0,3,5]
        position_history[:,ids_to_remove] = 0
        ids_to_keep = invertInds(position_history.shape[1], ids_to_remove)
        history_with_removal = position_history[:,ids_to_keep,:] 
        expected_history_with_removal = np.ones((100,5,2))
        self.assertTrue(np.all(history_with_removal==expected_history_with_removal))

def slicing_test():
    # 10x90
    arr = np.ones((5,20))
    # Get rid of columns
    arr[:,[3,5]] = 0
    print(arr)
    arr2 = np.ones((5,20))
    id_list = [3,5]
    arr2[:, id_list] = 0
    print(arr2)


if __name__ == '__main__':
    unittest.main()
