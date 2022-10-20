'''
Defines a class that provides simple methods for solving the Schrödinger
equation (SE) in coordinate space.
'''
import numpy as np
import numpy.typing as npt
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
from scipy.misc import derivative

from .interaction import Interaction
from .free_solutions import phase_shift

# Default values for solving the SE.
DEFAULT_R_MIN = 1e-6 # fm
DEFAULT_R_MAX = 50.0 # fm
DEFAULT_R_0 = 20.0 # fm
DEFAULT_NUM_PTS = 10000
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
        energy: float, # E_{c.m.} (MeV)
        args: npt.ArrayLike, # interaction parameters
        s_mesh: npt.ArrayLike, # s where phi(s) is calculated
        l: int = 0, # angular momentum
        phi_0: float = 0.0, # phi(r=0)
        phi_prime_0: float = 1.0, # dphi/dr(r=0)
        max_steps: int = MAX_STEPS, # see scipy.integrate.odeint documentation
        return_uprime: bool = False # Return u'(r) as well as u(r)
    ):
        '''
        Solves the Schrödinger equation at the specified center-of-mass energy.
        Returns a 2-column matrix. The first is the r values. The second is the
        reduced radial wavefunction, u(r). (The optional third - based on
        return_uprime - is u'(r).)
        '''
        initial_conditions = np.array([phi_0, phi_prime_0]) # initial phi(0) and phi'(0) conditions
        sol = odeint(
            lambda phi, s: np.array([phi[1],
                (self.interaction.tilde(s, args, energy) + l*(l+1)/s**2 - 1.0) * phi[0]]),
            initial_conditions, s_mesh, mxstep=max_steps, atol=1e-12, rtol=1e-12
        )

        if return_uprime:
            return np.vstack((s_mesh, sol)).T
        else:
            return np.vstack((s_mesh, sol[:, 0])).T
    

    def solve_se2(self,
        energy: float, # E_{c.m.} (MeV)
        args: npt.ArrayLike, # interaction parameters
        s_mesh: npt.ArrayLike, # s where phi(s) is calculated
        l: int = 0, # angular momentum
        phi_0: float = 0.0, # phi(r=0)
        phi_prime_0: float = 1.0, # dphi/dr(r=0)
        return_uprime: bool = False, # Return u'(r) as well as u(r)
        **solve_ivp_kwargs
    ):
        '''
        Solves the Schrödinger equation at the specified center-of-mass energy.
        Returns a 2-column matrix. The first is the r values. The second is the
        reduced radial wavefunction, u(r). (The optional third - based on
        return_uprime - is u'(r).)
        '''
        initial_conditions = np.array([phi_0, phi_prime_0]) # initial phi(0) and phi'(0) conditions
        sol = solve_ivp(
            lambda s, phi: np.array([phi[1],
                (self.interaction.tilde(s, args, energy) + l*(l+1)/s**2 - 1.0) * phi[0]]),
            s_mesh[[0, -1]], initial_conditions, **solve_ivp_kwargs
        )

        f = interp1d(sol.t, sol.y[0, :], kind='cubic')

        if return_uprime:
            fp = interp1d(sol.t, sol.y[1, :], kind='cubic')
            return np.vstack((s_mesh, f(s_mesh), fp(s_mesh))).T
        else:
            return np.vstack((s_mesh, f(s_mesh))).T


    def delta(self,
        energy: float, # center-of-mass energy
        args: npt.ArrayLike, # interaction parameters
        s_mesh: npt.ArrayLike, # s where phi(s) in calculated
        l: int, # angular momentum
        s_0: float, # phaseshift is extracted at phi(s_0)
        dx: float = 1e-6, # step size for the numerical derivative
        **kwargs # passed to solve_se
    ):
        '''
        Calculates the lth partial wave phase shift at the specified energy.
        kwargs are passed to solve_se
        '''
        solution = self.solve_se(energy, args, s_mesh, l, **kwargs)
        s = solution[:, 0]
        u = solution[:, 1]
        return phase_shift(u, s, l, s_0)


    def true_phi_solver(self,
        energy: float, # center-of-mass energy
        args: npt.ArrayLike, # interaction parameters
        s_mesh: npt.ArrayLike, # s where phi(s) in calculated
        l: int, # angular momentum
        **kwargs # passed to solve_se
    ):
        solution = self.solve_se(energy, args, s_mesh, l, **kwargs)
        return solution[:, 1]