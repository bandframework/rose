from typing import Callable

import numpy as np 
import numpy.typing as npt
from scipy.interpolate import interp1d
from mpmath import coulombf

from .schroedinger import SchroedingerEquation
from .free_solutions import phi_free

class Basis:
    def __init__(self,
        solver: SchroedingerEquation,
        theta_train: np.array, # training space
        s_mesh: np.array, # s = kr; discrete mesh where phi(s) is calculated
        n_basis: int, # number of basis vectors
        energy: float, # MeV, c.m.
        l: int # orbital angular momentum
    ):
        self.solver = solver
        self.theta_train = np.copy(theta_train)
        self.s_mesh = np.copy(s_mesh)
        self.n_basis = n_basis
        self.energy = energy
        self.l = l
    

    def phi_hat(self, coefficients):
        '''
        Every basis should know how to reconstruct hat{phi} from a set of
        coefficients. However, this is going to be different for each basis, so
        we will leave it up to the subclasses to implement this.
        '''
        raise NotImplementedError
    

class RelativeBasis(Basis):
    def __init__(self,
        solver: SchroedingerEquation,
        theta_train: np.array, # training space
        s_mesh: np.array, # s = kr; discrete mesh where phi(s) is calculated
        n_basis: int, # number of basis vectors
        energy: float, # MeV, c.m.
        l: int, # orbital angular momentum
        use_svd: bool # use principal components?
    ):
        super().__init__(solver, theta_train, s_mesh, n_basis, energy, l)

        # Returns Bessel functions when eta = 0.
        self.phi_0 = np.array([coulombf(self.l, self.solver.interaction.eta, rho) for rho in self.s_mesh], dtype=np.float64)

        self.all_vectors = np.array([
            self.solver.phi(energy, theta, self.s_mesh, l) - self.phi_0 for theta in theta_train
        ]).T

        if use_svd:
            U, S, _ = np.linalg.svd(self.all_vectors, full_matrices=False)
            self.singular_values = np.copy(S)
            self.all_vectors = np.copy(U)
        
        self.vectors = np.copy(self.all_vectors[:, :self.n_basis])
    

    def phi_hat(self, coefficients):
        return self.phi_0 + np.sum(coefficients * self.vectors, axis=1)


class CustomBasis(Basis):
    def __init__(self,
        solutions: np.array, # HF solutions, columns
        phi_0: np.array, # "offset", generates inhomogeneous term
        rho_mesh: np.array, # rho mesh; MUST BE EQUALLY SPACED POINTS!!!
        n_basis: int,
        # energy: float,
        # l: int,
        use_svd: bool
    ):
        self.solutions = solutions.copy()
        self.rho_mesh = rho_mesh.copy()
        self.n_basis = n_basis
        # self.energy = energy
        # self.ell = l

        self.phi_0 = phi_0.copy()

        if use_svd:
            U, S, _ = np.linalg.svd(self.solutions, full_matrices=False)
            self.singular_values = S.copy()
            self.solutions = U.copy()
        else:
            self.singular_values = None
        
        self.vectors = self.solutions[:, :self.n_basis].copy()
        
        # interpolating functions
        # To extrapolate or not to extrapolate?
        self.phi_0_interp = interp1d(self.rho_mesh, self.phi_0, kind='cubic')
        self.vectors_interp = [interp1d(self.rho_mesh, row, kind='cubic') for row in self.vectors.T]
    

    def phi_hat(self, coefficients):
        return self.phi_0 + np.sum(coefficients * self.vectors, axis=1)
    

    def interpolate_phi_0(self, rho):
        return self.phi_0_interp(rho)
    

    def interpolate_vectors(self, rho):
        return np.array(
            [f(rho) for f in self.vectors_interp]
        )