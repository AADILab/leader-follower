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

def randomPositions(high_bounds: NDArray[np.float64], num_positions: int, low_bounds: NDArray[np.float64] = np.array([0.,0.])) -> NDArray[np.float64]:
    """Generate an array of random positions according to the given constraints"""
    return np.hstack((
                np.random.uniform(low_bounds[0], high_bounds[0], size=(num_positions,1)),
                np.random.uniform(low_bounds[1], high_bounds[1], size=(num_positions,1))
            ))

def boundAnglePiToPi(heading):
    bounded_heading = heading
    # Bound heading from [0,2pi]
    if bounded_heading > 2*np.pi or bounded_heading < 0:
        bounded_heading %= 2*np.pi
    # Bound heading from [-pi,+pi]
    if bounded_heading > np.pi:
        bounded_heading -= 2*np.pi
    return bounded_heading

def calculateCentroid(positions):
    if positions.size == 0:
        return None
    else:
        return np.average(positions, axis=0)

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]
