'''
Defines a class that provides simple methods for solving the Schrödinger
equation (SE) in coordinate space.
'''
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from .interaction import Interaction
from .free_solutions import H_minus, H_plus, H_minus_prime, H_plus_prime

# Default values for solving the SE.
DEFAULT_R_MIN = 1e-12 # fm
DEFAULT_R_MAX = 30.0 # fm
DEFAULT_R_0 = 20.0 # fm
DEFAULT_NUM_PTS = 2000
MAX_STEPS = 20000


class SchroedingerEquation:
    def __init__(self,
        interaction: Interaction
    ):
        '''
        Instantiates an object that stores the Interaction and makes it easy to
        compute solutions to the Schrödinger equation and extract phase shifts
        with that Interaction.
        '''
        self.interaction = interaction


    def solve_se(self,
        energy: float, # E_{c.m.} (MeV)
        args: npt.ArrayLike, # interaction parameters
        s_endpts: npt.ArrayLike, # s where phi(s) is calculated
        l: int = 0, # angular momentum
        phi_0: float = 0.0, # phi(r=0)
        phi_prime_0: float = 1.0, # dphi/dr(r=0)
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
            s_endpts, initial_conditions, rtol=1e-12, atol=1e-12,
            dense_output=True, **solve_ivp_kwargs
        )
        return sol.sol


    def delta(self,
        energy: float, # center-of-mass energy
        args: npt.ArrayLike, # interaction parameters
        s_endpts: npt.ArrayLike, # [s_min, s_max]; phi(s) is calculated on this interval 
        l: int, # angular momentum
        s_0: float, # phaseshift is extracted at phi(s_0)
        **solve_ivp_kwargs # passed to solve_se
    ):
        '''
        Calculates the lth partial wave phase shift at the specified energy.
        kwargs are passed to solve_se
        '''
        solution = self.solve_se(energy, args, s_endpts, l=l, **solve_ivp_kwargs)
        u = solution(s_0)
        rl = 1/s_0 * (u[0]/u[1])
        return np.log(
            (H_minus(s_0, l) - s_0*rl*H_minus_prime(s_0, l)) / 
            (H_plus(s_0, l) - s_0*rl*H_plus_prime(s_0, l))
        ) / 2j


    def phi(self,
        energy: float, # center-of-mass energy
        args: npt.ArrayLike, # interaction parameters
        s_mesh: npt.ArrayLike, # s where phi(s) in calculated
        l: int, # angular momentum
        s_min: float = DEFAULT_R_MIN, # What do we call "zero"?
        solve_se_dict: dict = {}, # Options for solve_se: phi_0 and phi_prime_0
        **solve_ivp_kwargs # passed to solve_se
    ):
        '''
        Computes phi(s_mesh)
        '''
        solution = self.solve_se(energy, args, [s_min, s_mesh[-1]], l, **solve_se_dict, **solve_ivp_kwargs)
        return solution(s_mesh)[0, :]
    

    def phi_normalized(self,
        energy: float, # center-of-mass energy
        args: npt.ArrayLike, # interaction parameters
        s_mesh: npt.ArrayLike, # s where phi(s) in calculated
        l: int, # angular momentum
        **solve_ivp_kwargs # passed to solve_se
    ):
        '''
        Computes phi(s_mesh), but with max(phi(s_mesh)) = 1.
        '''
        phi = self.phi(energy, args, s_mesh, l, **solve_ivp_kwargs)
        return phi / np.max(np.abs(phi))

