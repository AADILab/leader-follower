import numpy as np
from numpy.typing import NDArray

def calculateDistance(positions_a: NDArray[np.float64], positions_b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the distance between positions A and B"""
    return np.linalg.norm(positions_a-positions_b, axis=1)

def calculateDeltaHeading(current_heading: float, desired_heading: float) -> float:
    """ Calculate delta headings such that delta is the shortest path from
    current heading to the desired heading.
    """
    if desired_heading == current_heading:
        delta_heading = 0
    else:
        # Case 1: Desired heading greater than current heading
        if desired_heading > current_heading:
            desired_heading_prime = desired_heading - 2*np.pi

        # Case 2: Desired heading less than current heading
        else:
            desired_heading_prime = desired_heading + 2*np.pi

        delta0 = desired_heading - current_heading
        delta1 = desired_heading_prime - current_heading
        which_delta = np.argmin([np.abs(delta0), np.abs(delta1)])
        delta_heading = np.array([delta0, delta1])[which_delta]
    return delta_heading
