'''
Defines a general template for interactions.
Includes some "hard-coded" interactions.
'''
from typing import Callable
import numpy as np

from .constants import HBARC

class Interaction:
    '''
    Defines a local interaction.

    '''
    def __init__(self,
        coordinate_space_potential: Callable[[float, np.ndarray], float],
        n_theta: int,
        mu: float,
        energy: float,
        is_complex: bool = False
    ):
        '''
        Instantiates a local interaction.

        :param coordinate_space_potential: V(r, theta)
        :param n_theta: How many parameters does the interaction have? (size of theta)
        :param mu: reduced mass (MeV)
        :param energy: center-of-mass energy
        :param is_complex: Is the interaction complex?
        :return: local interaction object
        :rtype: Interaction

        '''
        self.v_r = coordinate_space_potential
        self.n_theta = n_theta
        self.mu = mu / HBARC # Go ahead and convert to 1/fm
        self.energy = energy
        self.is_complex = is_complex


    def tilde(self,
        s: float,
        alpha: np.ndarray
    ):
        '''
        Computes tilde{U}(s, alpha, E) = V(s, alpha) / E
        E = E_{c.m.}, [E] = MeV = [v_r]

        :param s: s = rho = kr (dimensionless)
        :param alpha: interaction parameters
        :return: value of interaction at parameter point alpha on grid point s
        :rtype: float

        '''
        p = np.sqrt(2*self.mu*self.energy/HBARC) # 1/fm
        return  1.0/self.energy * self.v_r(s/p, alpha)


    def basis_functions(self,
        rho_mesh: np.ndarray
    ):
        '''
        Computes the "bare" potential.
        Assumes tilde{U}(s, alpha) is affine in "all" of the alpha components.

        :param rho_mesh: rho = s = kr (dimensionless)
        :return: matrix of columns; each corresponding to the "bare" component (alpha[column] = 1)
        :rtype: numpy.ndarray

        '''
        return np.array([
            self.tilde(rho_mesh, row) for row in np.eye(self.n_theta)
        ]).T
    

    def coefficients(self,
        alpha: np.ndarray # interaction parameters
    ):
        '''
        Basis function coefficients.
        For affine potentials, this is just the interaction parameters.

        :param alpha: interaction parameters
        :return: basis function coefficients
        :rtype: numpy.ndarray

        '''
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