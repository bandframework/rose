'''
Defines a class to package the spin-orbit term.
'''

from typing import Callable
import numpy as np

class SpinOrbitTerm():
    def __init__(self,
        spin_orbit_potential: Callable[[float, np.array, float], float], #V_{SO}(r, theta, l•s)
        l_dot_s: float # l•s
    ):
        self.l_dot_s = l_dot_s
        self.spin_orbit_potential = lambda r, alpha: spin_orbit_potential(r, alpha, self.l_dot_s)
