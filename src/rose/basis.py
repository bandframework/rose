import pickle
from typing import Callable

import numpy as np 
import numpy.typing as npt
from scipy.interpolate import interp1d
from mpmath import coulombf

from .constants import HBARC
from .schroedinger import SchroedingerEquation
from .energized_interaction_eim import EnergizedInteractionEIM

class Basis:
    '''Base class / template
    '''

    @classmethod
    def load(cls, filename):
        r'''Loads a previously saved Basis.
        
        '''
        with open(filename, 'rb') as f:
            basis = pickle.load(f)
        return basis


    def __init__(self,
        solver: SchroedingerEquation,
        theta_train: np.array,
        rho_mesh: np.array,
        n_basis: int,
        l: int
    ):
        r'''Builds a reduced basis.
        
        Parameters:
            solver (SchroedingerEquation): high-fidelity solver
            theta_train (ndarray): training space
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            l (int): orbital angular momentum
        
        Attributes:
            solver (SchroedingerEquation): high-fidelity solver
            theta_train (ndarray): training space
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            l (int): orbital angular momentum

        '''
        self.solver = solver
        self.theta_train = np.copy(theta_train)
        self.rho_mesh = np.copy(rho_mesh)
        self.n_basis = n_basis
        self.l = l
    

    def phi_hat(self, coefficients):
        r'''Emulated wave function.
        
        Every basis should know how to reconstruct hat{phi} from a set of
        coefficients. However, this is going to be different for each basis, so
        we will leave it up to the subclasses to implement this.

        Parameters:
            coefficients (ndarray): expansion coefficients
        
        Returns:
            phi_hat (ndarray): approximate wave function

        '''
        raise NotImplementedError
    

    def save(self, filename):
        '''Saves a basis to file.
        
        Parameters:
            filename (string): name of file

        '''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    

class RelativeBasis(Basis):
    def __init__(self,
        solver: SchroedingerEquation,
        theta_train: np.array,
        rho_mesh: np.array,
        n_basis: int,
        l: int,
        use_svd: bool,
        phi_0_energy: float = None
    ):
        r'''Builds a "relative" reduced basis. This is the default choice.

        $$
        \phi_{\rm HF} \approx \hat{\phi} = \phi_0 + \sum_i c_i \tilde{\phi}_i~.
        $$
        
        Parameters:
            solver (SchroedingerEquation): high-fidelity solver
            theta_train (ndarray): training space
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            l (int): orbital angular momentum
            use_svd (bool): Use principal components for $\tilde{\phi}$?
            phi_0_energy (float): energy at which $\phi_0$ is calculated
        
        Attributes:
            solver (SchroedingerEquation): high-fidelity solver
            theta_train (ndarray): training space
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            l (int): orbital angular momentum
            phi_0 (ndarray): free solution (no interaction)
            all_vectors (ndarray): HF solutions minus the free solution (one for
                each column)
            pillars (ndarray): $\tilde{\phi}_i$
            singular_values (ndarray): singular values from SVD
            vectors (ndarray): copy of `pillars`

        '''

        super().__init__(solver, theta_train, rho_mesh, n_basis, l)

        if phi_0_energy:
            k = np.sqrt(2*self.solver.interaction.mu*phi_0_energy/HBARC)
            eta = self.solver.interaction.k_c / k
        else:
            if isinstance(self.solver.interaction, EnergizedInteractionEIM):
                # k_mean = np.sqrt(2*self.solver.interaction.mu*np.mean(theta_train[:, 0])/HBARC)
                # eta = self.solver.interaction.k_c / k_mean
                # Does not support Coulomb (yet).
                eta = 0.0
            else:
                # If the phi_0 energy is not specified, we're only going to work
                # with non-Coulombic systems (for now).
                eta = 0.0
        
        # Returns Bessel functions when eta = 0.
        self.phi_0 = np.array([coulombf(self.l, eta, rho) for rho in self.rho_mesh], dtype=np.float64)

        self.all_vectors = np.array([
            self.solver.phi(theta, self.rho_mesh, l) - self.phi_0 for theta in theta_train
        ]).T
        self.pillars = self.all_vectors.copy()

        if use_svd:
            U, S, _ = np.linalg.svd(self.all_vectors, full_matrices=False)
            self.singular_values = np.copy(S)
            self.pillars = np.copy(U)
        
        self.vectors = np.copy(self.pillars[:, :self.n_basis])
    

    def phi_hat(self, coefficients):
        r'''Emulated wave function.
        
        Parameters:
            coefficients (ndarray): expansion coefficients
        
        Returns:
            phi_hat (ndarray): approximate wave function

        '''
        return self.phi_0 + np.sum(coefficients * self.vectors, axis=1)


class CustomBasis(Basis):
    def __init__(self,
        solutions: np.array, # HF solutions, columns
        phi_0: np.array, # "offset", generates inhomogeneous term
        rho_mesh: np.array, # rho mesh; MUST BE EQUALLY SPACED POINTS!!!
        n_basis: int,
        ell: int, # angular momentum, l
        use_svd: bool
    ):
        r'''Builds a custom basis. Allows the user to supply their own.

        $$
        \phi_{\rm HF} \approx \hat{\phi} = \phi_0 + \sum_i c_i \tilde{\phi}_i~.
        $$
        
        Parameters:
            solutions (ndarray): HF solutions
            phi_0 (ndarray): free solution (no interaction)
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            ell (int): orbital angular momentum
            use_svd (bool): Use principal components for $\tilde{\phi}$?
        
        Attributes:
            solver (SchroedingerEquation): not specified or assumed at construction
            theta_train (ndarray): not specified or assumed at construction
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            ell (int): orbital angular momentum
            phi_0 (ndarray): free solution (no interaction)
            solutions (ndarray): HF solutions provided by the user
            pillars (ndarray): $\tilde{\phi}_i$
            singular_values (ndarray): singular values from SVD
            vectors (ndarray): copy of `pillars`
            phi_0_interp (interp1d): interpolating function for $\phi_0$
            vectors_interp (interp1d): interpolating functions for vectors (basis states)

        '''

        super().__init__(None, None, rho_mesh, n_basis, ell)

        self.solutions = solutions.copy()
        self.pillars = solutions.copy()
        self.rho_mesh = rho_mesh.copy()
        self.n_basis = n_basis
        # Energy and angular momentum are not actually used, but it may be
        # useful to store them here to help the user keep track of which
        # CustomBasis this is.
        self.ell = ell

        self.phi_0 = phi_0.copy()

        if use_svd:
            U, S, _ = np.linalg.svd(self.solutions - self.phi_0[:, np.newaxis], full_matrices=False)
            self.singular_values = S.copy()
            self.pillars = U.copy()
        else:
            self.singular_values = None
        
        self.vectors = self.pillars[:, :self.n_basis].copy()
        
        # interpolating functions
        # To extrapolate or not to extrapolate?
        self.phi_0_interp = interp1d(self.rho_mesh, self.phi_0, kind='cubic')
        self.vectors_interp = [interp1d(self.rho_mesh, row, kind='cubic') for row in self.vectors.T]
    

    def phi_hat(self, coefficients):
        r'''Emulated wave function.
        
        Parameters:
            coefficients (ndarray): expansion coefficients
        
        Returns:
            phi_hat (ndarray): approximate wave function

        '''
        return self.phi_0 + np.sum(coefficients * self.vectors, axis=1)