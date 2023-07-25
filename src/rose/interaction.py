'''Wraps the user-defined interaction into a class that stores several relevant
parameters of the problem.
'''

from typing import Callable
import numpy as np
from .constants import HBARC, ALPHA
from .spin_orbit import SpinOrbitTerm

class Interaction:
    '''Defines a local, (possibly) complex, affine, fixed-energy interaction.
    '''
    def __init__(self,
        coordinate_space_potential: Callable[[float, np.array], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        energy: float, # E_{c.m.}
        ell: int,
        Z_1: int = 0, # atomic number of particle 1
        Z_2: int = 0, # atomic number of particle 2
        R_C: float = 0.0, # Coulomb "cutoff"
        is_complex: bool = False,
        spin_orbit_term: SpinOrbitTerm = None,
    ):
        r'''Creates a local, (possibly) complex, affine, fixed-energy interaction.

        Parameters:
            coordinate_space_potential (Callable[[float,ndarray],float]): V(r, theta)
            n_theta (int): number of parameters
            mu (float): reduced mass
            energy (float): center-of-mass, scattering energy
            ell (int): angular momentum
            Z_1 (int): charge of particle 1
            Z_2 (int): charge of particle 2
            R_C (float): Coulomb "cutoff" radius
            is_complex (bool): Is the interaction complex?
            spin_orbit_term (SpinOrbitTerm): See [Spin-Orbit section](#spin-orbit).
        
        Returns:
            instance (Interaction): instance of `Interaction`

        Attributes:
            v_r (Callable[[float,ndarray],float]): coordinate-space potential; $V(r, \alpha)$
            n_theta (int): number of parameters
            mu (float): reduced mass
            ell (int): angular momentum
            k_c (float): Coulomb momentum; $k\eta$
            is_complex (bool): Is this a complex potential?
            spin_orbit_term (SpinOrbitTerm): See [Spin-Orbit section](#spin-orbit)

        '''
        self.v_r = coordinate_space_potential
        self.n_theta = n_theta
        self.mu = mu / HBARC # Go ahead and convert to 1/fm
        self.ell = ell
        self.k_c = ALPHA * Z_1*Z_2 * self.mu
        self.R_C = R_C
        # self.eta = ALPHA * Z_1 * Z_2 * self.mu / self.k
        self.is_complex = is_complex
        self.spin_orbit_term = spin_orbit_term

        if spin_orbit_term is None:
            self.include_spin_orbit = False
        else:
            self.include_spin_orbit = True

        if energy:
            # If the energy is specified (not None as it is when subclass
            # EnergizedInteraction instantiates), set up associated attributes.
            self.energy = energy
            self.k = np.sqrt(2 * self.mu * self.energy/HBARC)
            self.sommerfeld = self.k_c / self.k
        else:
            # If the energy is not specified, these will be set up when the
            # methods are called.
            self.energy = None
            self.k = None
            self.sommerfeld = None


    def tilde(self,
        s: float,
        alpha: np.array
    ):
        r'''Scaled potential, $\tilde{U}(s, \alpha, E)$.

        * Does not include the Coulomb term.
        * $E = E_{c.m.}$; `[E] = MeV = [v_r]`

        Parameters:
            s (float): mesh point; $s = pr/\hbar$
            alpha (ndarray): the varied parameters
        
        Returns:
            u_tilde (float | complex): value of scaled interaction

        '''
        vr = self.v_r(s/self.k, alpha)
        vr += self.spin_orbit_term.spin_orbit_potential(s/self.k, alpha) if self.include_spin_orbit else 0
        return  1.0/self.energy * vr


    def basis_functions(self,
        rho_mesh: np.array
    ):
        r'''In general, we approximate the potential as

        $\hat{U} = \sum_{j} \beta_j(\alpha) u_j$
        
        For affine interactions (like those defined in this class) the basis
        functions (or "pillars), $u_j$, are just the "naked" parts of the
        potential. As seen below, it is assumed that the $\beta_j(\alpha)$
        coefficients are just the affine parameters, $\alpha$, themselves.

        Parameters:
            rho_mesh (ndarray): discrete $\rho$ values at which the potential is
                going to be evaluated
        
        Returns:
            value (ndarray): values of the scaled potential at provided $\rho$ points
        '''
        return np.array([
            self.tilde(rho_mesh, row) for row in np.eye(self.n_theta)
        ]).T
    

    def coefficients(self,
        alpha: np.array # interaction parameters
    ):
        r'''As noted in the `basis_functions` documentation, the coefficients
        for affine interactions are simply the parameter values. The inverse of
        the momentum is also returned to support energy emulation.

        Parameters:
            alpha (ndarray): parameter point
        
        Returns:
            result (tuple): inverse momentum and coefficients

        '''
        return 1/self.k, alpha
    

    def eta(self,
        alpha: np.array
    ):
        r'''Sommerfeld parameter. Implemented as a function to support energy
        emulation (where the energy could be a part of the parameter vector,
        `alpha`).
        
        Parameters:
            alpha (ndarray): parameter vector
        
        Returns:
            eta (float): Sommerfeld parametere
        '''
        return self.sommerfeld
    

    def momentum(self, alpha: np.array):
        r'''Momentum. Implemented as a function to support energy emulation
        (where the energy/momentum could be a part of the parameter vector,
        `alpha`).
        
        Parameters:
            alpha (ndarray): parameter vector
        
        Returns:
            k (float): center-of-mass, scattering momentum
        '''
        return self.k


class InteractionSpace:
    def __init__(self,
        coordinate_space_potential: Callable[[float, np.array], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        energy: float, # E_{c.m.}
        l_max: int,
        Z_1: int = 0, # atomic number of particle 1
        Z_2: int = 0, # atomic number of particle 2
        R_C: float = 0.0,
        is_complex: bool = False,
        spin_orbit_potential: Callable[[float, np.array, float], float] = None #V_{SO}(r, theta, l•s)
    ):
        r'''Generates a list of $\ell$-specific interactions.
        
        Parameters:
            coordinate_space_potential (Callable[[float,ndarray],float]): V(r, theta)
            n_theta (int): number of parameters
            mu (float): reduced mass
            energy (float): center-of-mass, scattering energy
            l_max (int): maximum angular momentum
            Z_1 (int): charge of particle 1
            Z_2 (int): charge of particle 2
            R_C (float): Coulomb "cutoff" radius
            is_complex (bool): Is the interaction complex?
            spin_orbit_potential (Callable[[float,ndarray,float],float]): coordinate-space, spin-orbit potential; $V_{\rm SO}(s, \alpha, 2\ell\cdot s$)
        
        Returns:
            instance (InteractionSpace): instance of InteractionSpace
        
        Attributes:
            interaction (list): list of `Interaction`s
        '''
        self.interactions = []
        if spin_orbit_potential is None:
            for l in range(l_max+1):
                self.interactions.append(
                    [Interaction(coordinate_space_potential, n_theta, mu,
                        energy, l, Z_1=Z_1, Z_2=Z_2, R_C=R_C,
                        is_complex=is_complex)]
                )
        else:
            for l in range(l_max+1):
                self.interactions.append(
                    [Interaction(coordinate_space_potential, n_theta, mu,
                        energy, l, Z_1=Z_1, Z_2=Z_2, R_C=R_C,
                        is_complex=is_complex,
                        spin_orbit_term=SpinOrbitTerm(spin_orbit_potential,
                        lds)) for lds in couplings(l)]
                )


def couplings(l):
    r'''For a spin-1/2 nucleon scattering off a spin-0 nucleus, there are
    maximally 2 different total angular momentum couplings: l+1/2 and l-1/2.
    
    Parameters:
        l (int): angular momentum
    
    Returns:
        couplings (list): total angular momentum possibilities
    '''
    if l == 0:
        return [l]
    else:
        return [l, -(l+1)]

