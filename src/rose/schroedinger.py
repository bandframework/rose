r'''`SchroedingerEquation` is a high-fidelity (HF), Schrödinger-equation solver for
local, complex interactions.

By default, `rose` will provide HF solution using `scipy.integrate.solve_ivp`.
For details about providing your own solutions, see [Basis
documentation](basis.md).

'''
import numpy as np
from scipy.integrate import solve_ivp

from .interaction import Interaction
from .free_solutions import H_minus, H_plus, H_minus_prime, H_plus_prime
from .utility import regular_inverse_s

# Default values for solving the SE.
DEFAULT_R_MIN = 1e-12 # fm
DEFAULT_R_MAX = 30.0 # fm
DEFAULT_R_0 = 20.0 # fm
DEFAULT_NUM_PTS = 2000
MAX_STEPS = 20000
PHI_THRESHOLD = 1e-10

class SchroedingerEquation:
    '''
    High-fidelity (HF) solver for optical potentials. How high-fidelity? You decide!
    '''
    def __init__(self,
        interaction: Interaction,
        hifi_tolerances: list = [1e-12, 1e-12]
    ):
        r'''Solves the Shrödinger equation for local, complex potentials.
        
        Parameters:
            interaction (Interaction): See [Interaction documentation](interaction.md).
            hifi_tolerances (list): 2-element list of numbers: the relative
                tolerance, `rel_tol`, and the absolute tolerance `abs_tol`

        Returns:
            solver (SchroedingerEquation): instance of `SchroedingerEquation`

        '''
        self.interaction = interaction
        self.rel_tol = hifi_tolerances[0]
        self.abs_tol = hifi_tolerances[1]


    def solve_se(self,
        alpha: np.array, # interaction parameters
        s_endpts: np.array, # s where phi(s) is calculated
        l: int = 0, # angular momentum
        rho_0 = None, # initial rho value ("effective zero")
        phi_threshold = PHI_THRESHOLD, # minimum phi value (zero below this value)
        **solve_ivp_kwargs
    ):
        r'''Solves the reduced, radial Schrödinger equation.
        
        Parameters:
            alpha (ndarray): parameter vector
            s_endpts (ndarray): lower and upper bounds of the $s$ mesh.
            l (int): angular momentum
            rho_0 (float): initial $\rho$ (or $s$) value; starting point for the
                solver
            phi_threshold (float): minimum $\phi$ value; The wave function is
                considered zero below this value.
        
        Returns:
            array (ndarray): 2-column matrix; The first column is the $r$ values.
                The second is the reduced radial wavefunction, $u(r)$. (The optional
                third - based on return_uprime - is $u^\prime(r)$.)
        '''

        C_l = Gamow_factor(l, self.interaction.eta(alpha))
        S_C = self.interaction.momentum(alpha) * self.interaction.R_C

        if rho_0 is None:
            rho_0 = (phi_threshold / C_l) ** (1/(l+1))
        phi_0 = C_l * rho_0**(l+1)
        phi_prime_0 = C_l * (l+1) * rho_0**l
        
        if self.interaction.is_complex:
            initial_conditions = np.array([phi_0+0j, phi_prime_0+0j])
        else:
            initial_conditions = np.array([phi_0, phi_prime_0])
        
        sol = solve_ivp(
            lambda s, phi: np.array([phi[1],
                (self.interaction.tilde(s, alpha) + \
                 2*self.interaction.eta(alpha) * regular_inverse_s(s, S_C) + l*(l+1)/s**2 - 1.0) * phi[0]]),
            s_endpts, initial_conditions, rtol=self.rel_tol, atol=self.abs_tol,
            dense_output=True, **solve_ivp_kwargs
        )

        return sol.sol


    def delta(self,
        alpha: np.array, # interaction parameters
        s_endpts: np.array, # [s_min, s_max]; phi(s) is calculated on this interval 
        l: int, # angular momentum
        s_0: float, # phaseshift is extracted at phi(s_0)
        **solve_ivp_kwargs # passed to solve_se
    ):
        r'''Calculates the $\ell$-th partial wave phase shift at the specified energy.
        solve_ivp_kwargs are passed to solve_se

        Parameters:
            alpha (ndarray): parameter vector
            s_endpts (ndarray): lower and upper bounds of the $s$ mesh.
            l (int): angular momentum
            s_0 (float): $s$ value where the phase shift is calculated (must be
                less than the second element in `s_endpts`)
        
        Returns:
            delta (float): phase shift extracted from the reduced, radial
                wave function

        '''
        # Should s_endpts be [s_min, s_endpts[1]]?
        solution = self.solve_se(alpha, s_endpts, l=l, **solve_ivp_kwargs)
        u = solution(s_0)
        rl = 1/s_0 * (u[0]/u[1])
        return np.log(
            (H_minus(s_0, l, self.interaction.eta(alpha)) - s_0*rl*H_minus_prime(s_0, l, self.interaction.eta(alpha))) / 
            (H_plus(s_0, l, self.interaction.eta(alpha)) - s_0*rl*H_plus_prime(s_0, l, self.interaction.eta(alpha)))
        ) / 2j


    def phi(self,
        alpha: np.array, # interaction parameters
        s_mesh: np.array, # s where phi(s) in calculated
        l: int, # angular momentum
        rho_0: float = None, # What do we call "zero"?
        phi_threshold: float = PHI_THRESHOLD,
        **solve_ivp_kwargs # passed to solve_se
    ):
        r'''Computes the reduced, radial wave function $\phi$ (or $u$) on `s_mesh`.

        Parameters:
            alpha (ndarray): parameter vector
            s_mesh (ndarray): values of $s$ at which $\phi$ is calculated
            l (int): angular momentum
            rho_0 (float): starting point for the solver
            phi_threshold (float): minimum $\phi$ value; The wave function is
                considered zero below this value.
        
        Returns:
            phi (ndarray): reduced, radial wave function

        '''
        if rho_0 is None:
            rho_0 = (phi_threshold / Gamow_factor(l, self.interaction.eta(alpha))) ** (1/(l+1))

        solution = self.solve_se(alpha, [rho_0, s_mesh[-1]], l, rho_0=rho_0,
                                 phi_threshold=phi_threshold, **solve_ivp_kwargs)

        ii_0 = np.where(s_mesh < rho_0)[0]
        y = solution(s_mesh)[0]
        y[ii_0] = 0
        return y
    

def Gamow_factor(l, eta):
    r'''This returns the... Gamow factor.
    See [Wikipedia](https://en.wikipedia.org/wiki/Gamow_factor).

    Parameters:
        l (int): angular momentum
        eta (float): Sommerfeld parameter (see
            [Wikipedia](https://en.wikipedia.org/wiki/Sommerfeld_parameter))
    
    Returns:
        C_l (float): Gamow factor

    '''
    if eta == 0.0:
        if l == 0:
            return 1
        else:
            return 1 / (2*l + 1) * Gamow_factor(l-1, 0)
    elif l == 0:
        return np.sqrt(2*np.pi*eta / (np.exp(2*np.pi*eta)-1))
    else:
        return np.sqrt(l**2 + eta**2) / (l*(2*l+1)) * Gamow_factor(l-1, eta)
