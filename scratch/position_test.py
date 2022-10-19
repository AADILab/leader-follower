import numpy as np
from numpy.typing import NDArray

# class POIColonyState():
#     def __init__(self, positions) -> None:
#         self.positions = positions
#         self.num_pois = positions.shape[0]

# class POI():
#     def __init__(self, state: POIColonyState, id: int) -> None:
#         self.state = state
#         self.id = id

#     @property
#     def position(self) -> NDArray[np.float64]:
#         return self.state.positions[self.id]

# class POIColony():
#     def __init__(self, state: POIColonyState) -> None:
#         self.state = state
#         self.pois = [POI(self.state, id) for id in range(self.state.num_pois)]

# positions = np.random.uniform(size=(10,2))
# c = POIColony(state = POIColonyState(positions=positions))

# print(c.state.positions[0])
# print(c.pois[0].position)

# positions = np.random.uniform(size=(10,2))


class Single():
    def __init__(self, position: NDArray[np.float64]) -> None:
        self.position = position

class Colony():
    def __init__(self, positions: NDArray[np.float64]) -> None:
        self.positions = positions
        self.singles = [Single(position) for position in positions]

    def reset(self, positions: NDArray[np.float64]) -> None:
        self.positions[:,:] = positions

positions = np.random.uniform(size=(10,2))

c = Colony(positions)
print(positions[0])
print(c.singles[0].position)

new_positions = np.random.uniform(size=(10,2))
print(new_positions[0])
c.reset(new_positions)

print(c.singles[0].position)

