'''
Defines a class that provides simple methods for solving the Schrödinger
equation (SE) in coordinate space.
'''
import numpy as np
import numpy.typing as npt
from scipy.integrate import odeint

from .interaction import Interaction
from .constants import HBARC

# Default values for solving the SE.
DEFAULT_R_MIN = 1e-6 # fm
DEFAULT_R_MAX = 50 # fm
DEFAULT_NUM_PTS = 10000
DEFAULT_U_0 = 0.0
DEFAULT_UPRIME_0 = 1.0
MAX_STEPS = 20000

class SchroedingerEquation:
    def __init__(self,
        interaction: Interaction,
    ):
        '''
        Instantiates an object that stores the Interaction.
        '''
        self.interaction = interaction


    def solve_se(self,
        energy:float, # E_{c.m.} (MeV)
        args:npt.ArrayLike, # interaction parameters
        l: int = 0, # angular momentum
        u_0: float = DEFAULT_U_0, # u(r=0)
        up_0:float = DEFAULT_UPRIME_0, # du/dr(r=0)
        r_min: float = DEFAULT_R_MIN, # Starting r values in the shooting method.
        r_max: float = DEFAULT_R_MAX, # Largest r value to "integrate out" to.
        num_pts: int = DEFAULT_NUM_PTS,
        max_steps:int = MAX_STEPS, # see scipy.integrate.odeint documentation
        return_uprime:bool = False # Return u'(r) as well as u(r)
    ):
        '''
        Solves the Schrödinger equation at the specified center-of-mass energy.
        Returns a 2-column matrix. The first is the r values. The second is the
        reduced radial wavefunction, u(r). (The optional third - based on
        return_uprime - is u'(r).)
        '''
        s_values = np.linspace(r_min, r_max, num_pts) * np.sqrt(2*self.interaction.mu*energy/HBARC)
        initial_conditions = np.array([u_0, up_0]) # initial u and u' conditions

        sol = odeint(
            lambda phi, s: np.array([phi[1],
                (self.interaction.tilde(s, args, energy) + l*(l+1)/s**2 - 1.0) * phi[0]]),
            initial_conditions, s_values, mxstep=max_steps
        )

        if return_uprime:
            return np.vstack((s_values, sol)).T
        else:
            return np.vstack((s_values, sol[:, 0])).T