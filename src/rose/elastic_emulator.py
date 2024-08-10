import numpy as np
from numba import jit
from numba.experimental import jitclass


dtype = []


@jitclass(dtype)
class ElasticScatterEmulator:
    def __init__(self):
        pass
