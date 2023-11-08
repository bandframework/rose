"""Defines a class to package the spin-orbit term.
"""

from typing import Callable
import numpy as np


class SpinOrbitTerm:
    def __init__(
        self,
        spin_orbit_potential: Callable[
            [float, np.array, float], float
        ],  # V_{SO}(r, theta, l•s)
        l_dot_s: float,  # l•s
    ):
        r"""Spin-orbit interaction

        Parameters:
            spin_orbit_potential (Callable[[float, ndarray, float],float]):
                coordinate-space, spin-orbit potential
            l_dot_s (float): $2\ell\cdot s$ matrix elements, $+\ell$ or $-\ell-1$

        Attributes:
            l_dot_s (float): $2\ell\cdot s$ matrix elements, $+\ell$ or $-\ell-1$
            spin_orbit_potential: (Callable[[float, ndarray, float],float]):
                coordinate-space, spin-orbit potential

        """
        self.l_dot_s = l_dot_s
        self.v_so = spin_orbit_potential

    def spin_orbit_potential(r, alpha):
        return self.v_so(r, alpha, self.l_dot_s)
