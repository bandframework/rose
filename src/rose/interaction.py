'''
Defines a general template for interactions.
Includes some "hard-coded" interactions.
'''
from typing import Callable
import numpy as np
import numpy.typing as npt

from .constants import HBARC

class Interaction:
    '''
    Template class.
    '''
    def __init__(self,
        coordinate_space_potential: Callable[[float, npt.ArrayLike], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        energy: float, # E_{c.m.}
        is_complex: bool = False
    ):
        self.v_r = coordinate_space_potential
        self.n_theta = n_theta
        self.mu = mu / HBARC # Go ahead and convert to 1/fm
        self.energy = energy
        self.is_complex = is_complex


    def tilde(self,
        s: float,
        alpha: npt.ArrayLike
    ):
        '''
        tilde{U}(s, alpha, E)
        s = pr/hbar
        alpha are the parameters we are varying
        E = E_{c.m.}, [E] = MeV = [v_r]
        '''
        p = np.sqrt(2*self.mu*self.energy/HBARC) # 1/fm
        return  1.0/self.energy * self.v_r(s/p, alpha)


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
    True
)