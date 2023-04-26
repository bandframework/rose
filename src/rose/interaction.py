'''
Defines a general template for interactions.
Includes some "hard-coded" interactions.
'''
from typing import Callable
import numpy as np
import numpy.typing as npt

from .constants import HBARC, ALPHA

class Interaction:
    '''
    Template class.
    '''
    def __init__(self,
        coordinate_space_potential: Callable[[float, npt.ArrayLike], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        energy: float, # E_{c.m.}
        Z_1: int = 0, # atomic number of particle 1
        Z_2: int = 0, # atomic number of particle 2
        is_complex: bool = False
    ):
        self.v_r = coordinate_space_potential
        self.n_theta = n_theta
        self.mu = mu / HBARC # Go ahead and convert to 1/fm
        self.k_c = ALPHA * Z_1*Z_2 * self.mu
        # self.eta = ALPHA * Z_1 * Z_2 * self.mu / self.k
        self.is_complex = is_complex
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
        alpha: npt.ArrayLike
    ):
        '''
        tilde{U}(s, alpha, E)
        Does not include the Coulomb term.
        s = pr/hbar
        alpha are the parameters we are varying
        E = E_{c.m.}, [E] = MeV = [v_r]
        '''
        return  1.0/self.energy * self.v_r(s/self.k, alpha)


    def basis_functions(self,
        rho_mesh: npt.ArrayLike
    ):
        return np.array([
            self.tilde(rho_mesh, row) for row in np.eye(self.n_theta)
        ]).T
    

    def coefficients(self,
        alpha: npt.ArrayLike # interaction parameters
    ):
        return alpha
    

    def eta(self,
        alpha: np.array
    ):
        return self.sommerfeld


class EnergizedInteraction(Interaction):
    '''
    A version of Interaction that treats the energy as a parameter... kind of.
    '''
    def __init__(self, 
        coordinate_space_potential: Callable[[float, npt.ArrayLike], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        Z_1: int = 0, # atomic number of particle 1
        Z_2: int = 0, # atomic number of particle 2
        is_complex: bool = False
    ):
        # We'll initialize with energy = 0 (this should help find bugs) even
        # though the energy is not fixed.
        # n_theta includes the energy, which Interaction doesn't see as a
        # parameter, hence n_theta - 1.
        super().__init__(coordinate_space_potential, n_theta-1, mu, None,
            Z_1=Z_1, Z_2=Z_2, is_complex=is_complex)
    

    def tilde(self,
        s: float,
        alpha: np.array
    ):
        '''
        theta[0] is the energy
        '''
        energy = alpha[0]
        k = np.sqrt(2*self.mu*energy/HBARC)
        return  1.0/energy * self.v_r(s/k, alpha[1:])
    

    def basis_functions(self,
        rho_mesh: npt.ArrayLike
    ):
        return np.array([
            self.tilde(rho_mesh, np.hstack((1, row))) for row in np.eye(self.n_theta-1)
        ]).T
    

    def coefficients(self,
        alpha: npt.ArrayLike # interaction parameters
    ):
        '''
        alpha[0] is the energy
        '''
        return alpha[1:]


    def eta(self,
        alpha: np.array
    ):
        return self.k_c / np.sqrt(2*self.mu*alpha[0]/HBARC)
    

NUCLEON_MASS = 939.565 # neutron mass (MeV)
MU_NN = NUCLEON_MASS / 2 # reduced mass of the NN system (MeV)

def mn_potential(r, args):
    '''
    Minnesota potential
    '''
    v_0r, v_0s = args
    return v_0r * np.exp(-1.487*r**2) + v_0s*np.exp(-0.465*r**2)

# Stored instances of the Minnesota interaction for testing.
# Fixed at E_{c.m.} = 50 MeV.
MN_Potential = Interaction(
    mn_potential,
    2,
    MU_NN,
    50
)

def complex_mn_potential(r, args):
    vr, vi = args
    return mn_potential(r, [vr, 1j*vi])


Complex_MN_Potential = Interaction(
    complex_mn_potential,
    2,
    MU_NN,
    50,
    is_complex = True
)