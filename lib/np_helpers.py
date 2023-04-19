import numpy as np
from numpy.typing import NDArray

def invertInds(dimension, ids_to_remove):
    all_ids = list(range(dimension))
    ids_to_keep = []
    for id in all_ids:
        if id not in ids_to_remove:
            ids_to_keep.append(id)
    return ids_to_keep

