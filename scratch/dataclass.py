import numpy as np
from typing import List, Union, Optional

# Genome encodes weights of a network as list of numpy arrays
Genome = List[np.array]

class GenomeData():
    def __init__(self, genome: Genome, id: int, fitness: Optional[float]=None) -> None:
        self.genome = genome
        self.id = id
        self.fitness = fitness

    def __eq__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness == __o.fitness

    def __ne__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness != __o.fitness

    def __lt__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness < __o.fitness

    def __le__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness <= __o.fitness

    def __gt__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness > __o.fitness

    def __ge__(self, __o: object) -> bool:
        if __o.__class__ is self.__class__:
            return self.fitness >= __o.fitness

# gd = GenomeData(genome=[np.array([])],id=0)

population = [GenomeData(genome=[np.zeros(5)], id=id, fitness=id) for id in range(10)]

population.sort()

# np.array([0.0]) == "hello"

# print(gd == "hello")
print(id)