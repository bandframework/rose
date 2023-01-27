'''
Defines a class for "affinizing" Interactions using the Empirical Interpolation
Method (EIM).
'''

from typing import Callable
import numpy as np
import numpy.typing as npt
from scipy.stats import qmc

from .interaction import Interaction
from .constants import HBARC, DEFAULT_RHO_MESH

def max_vol(basis, indxGuess):
    # basis looks like a long matrix, the columns are the "pillars" V_i(x):
    #     [   V_1(x)
    #         V_2(x)
    #         .
    #         .
    #         .
    #     ]
    # indxGuess is a first guess of where we should "measure", or ask the questions
    nbases = basis.shape[1]
    interpBasis = np.copy(basis)

    for ij in range(len(indxGuess)):
        interpBasis[[ij,indxGuess[ij]],:] = interpBasis[[indxGuess[ij],ij],:]
    indexing = np.array(range(len(interpBasis)))

    for ij in range(len(indxGuess)):
        indexing[[ ij,indxGuess[ij] ]] = indexing[[ indxGuess[ij],ij ]]
    
    for iIn in range(1, 100):
        B = np.dot(interpBasis, np.linalg.inv(interpBasis[:nbases]))
        b = np.max(B)
        if b > 1:
            
            p1, p2 = np.where(B == b)[0][0], np.where(B == b)[1][0]
            interpBasis[[p1,p2],:] = interpBasis[[p2,p1],:]
            indexing[[p1,p2]] = indexing[[p2,p1]]
        else:
            break
        #this thing returns the indexes of where we should measure
    return np.sort(indexing[:nbases])


class InteractionEIM(Interaction):
    def __init__(self,
        coordinate_space_potential: Callable[[float, npt.ArrayLike], float], # V(r, theta)
        n_theta: int, # How many parameters does the interaction have?
        mu: float, # reduced mass (MeV)
        energy: float, # E_{c.m.}
        training_info: npt.ArrayLike,
        is_complex: bool = False,
        n_basis: int = None,
        explicit_training: bool = False,
        n_train: int = 20,
        rho_mesh: npt.ArrayLike = DEFAULT_RHO_MESH,
        match_points: np.array = None
    ):
        '''
        :param coordinate_space_potential: V(r, theta) where theta are the interaction parameters
        :param n_theta: number of interaction parameters
        :param mu: reduced mass (MeV); converted to 1/fm
        :param is_complex: Is the interaction complex (e.g. optical potentials)?
        :param n_basis: How many "columns" do we need?
        :param training_info: either (1) parameters bounds or (2) explicit training points
        if (1):
        This is a 2-column matrix. The first column are the lower
        bounds. The second are the upper bounds. Each row maps to a
        single parameter.
        if (2):
        This is MxN matrix. N is the number of parameters. M is the
        number of samples.
        explicit_training: Is training_info (1) or (2)? (1) is default
        :param n_train: How many snapshots to generate? Ignored if explicit_training is True.
        :param r: coordinate-space points at which the interaction is generated (used
        for training)

        '''

        super().__init__(coordinate_space_potential, n_theta, mu, energy, is_complex=is_complex)

        self.energy = energy

        '''
        Generate a basis used to approximate the potential.
        '''
        # Did the user specify the training points?
        if explicit_training:
            snapshots = np.array([self.tilde(rho_mesh, theta) for theta in training_info]).T
        else:
            # Generate training points using the user-provided bounds.
            sampler = qmc.LatinHypercube(d=len(training_info))
            sample = sampler.random(n_train)
            train = qmc.scale(sample, training_info[:, 0], training_info[:, 1])
            snapshots = np.array([self.tilde(rho_mesh, theta) for theta in train]).T
        
        U, S, _ = np.linalg.svd(snapshots, full_matrices=False)
        self.singular_values = np.copy(S)
        
        if match_points is None:
            if n_basis is None:
                n_basis = n_theta
            self.snapshots = np.copy(U[:, :n_basis])
            # random r points between 0 and 2π fm
            self.match_indices = max_vol(self.snapshots,
                np.random.randint(0, self.snapshots.shape[0], size=self.snapshots.shape[1]))
            self.match_points = rho_mesh[self.match_indices]
            self.r_i = np.copy(self.match_points)
        else:
            n_basis = match_points.size
            self.snapshots = np.copy(U[:, :n_basis])
            self.match_points = np.copy(match_points)
            self.match_indices = np.array([np.argmin(np.abs(rho_mesh - ri)) for ri in self.match_points])
            self.r_i = rho_mesh[self.match_indices]

        self.Ainv = np.linalg.inv(self.snapshots[self.match_indices])


    def coefficients(self,
        alpha: npt.ArrayLike
    ):
        u_true = self.tilde(self.r_i, alpha)
        return self.Ainv @ u_true


    def tilde_emu(self,
        s: float,
        alpha: npt.ArrayLike
    ):
        '''
        tilde{U}(s, alpha, E)
        s = pr/hbar
        alpha are the parameters we are varying
        E = E_{c.m.}, [E] = MeV = [v_r]

        '''
        x = self.coefficients(alpha)
        emu = np.sum(x * self.snapshots, axis=1)
        return emu
    

    def basis_functions(self, rho_mesh: npt.ArrayLike):
        return np.copy(self.snapshots)


def wood_saxon(r, R, a):
    return 1/(1 + np.exp((r-R)/a))


def wood_saxon_prime(r, R, a):
    return -1/a * np.exp((r-R)/a) / (1 + np.exp((r-R)/a))**2


def optical_potential(r, theta):
    Vv, Wv, Wd, Rv, av, Rd, ad = theta
    return -Vv * wood_saxon(r, Rv, av) - \
           1j*Wv * wood_saxon(r, Rv, av) - \
           1j*4*ad*Wd * wood_saxon_prime(r, Rd, ad)


NUCLEON_MASS = 939.565 # neutron mass (MeV)
MU_NN = NUCLEON_MASS / 2 # reduced mass of the NN system (MeV)

BOUNDS_VV = [1, 10]
BOUNDS_WV = [-10, 10]
BOUNDS_WD = [-10, 10]
BOUNDS_RV = [3, 5]
BOUNDS_AV = [0.8, 1.2]
BOUNDS_RD = [3, 5]
BOUNDS_AD = [0.8, 1.2]

BOUNDS = np.array([
    BOUNDS_VV,
    BOUNDS_WV,
    BOUNDS_WD,
    BOUNDS_RV,
    BOUNDS_AV,
    BOUNDS_RD,
    BOUNDS_AD
])

Optical_Potential = InteractionEIM(
    optical_potential,
    7,
    MU_NN,
    50,
    BOUNDS,
    is_complex = True
)