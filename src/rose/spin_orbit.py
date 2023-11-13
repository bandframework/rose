"""Defines a class to package the spin-orbit term.
"""
from numba import njit
from typing import Callable
import numpy as np


class SpinOrbitTerm:
    def __init__(
        self,
        spin_orbit_potential: Callable[
            [float, np.array, float], float
        ] = None,
        l_dot_s: float = None,
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

        if spin_orbit_potential is None:
            self.l_dot_s = 0
            self.v_so = null

    def spin_orbit_potential(self, r, alpha):
        return self.v_so(r, alpha, self.l_dot_s)


@njit
def null(r, alpha, l_dot_s):
    return 0.

