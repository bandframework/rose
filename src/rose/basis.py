import numpy as np 
from typing import Callable
import numpy.typing as npt

from .interaction import Interaction
from .schroedinger import SchroedingerEquation
from .free_solutions import phi_free
from .constants import HBARC

class Basis:
    def __init__(self,
        interaction: Interaction,
        theta_train: npt.ArrayLike, # training space
        s_mesh: npt.ArrayLike, # s = kr; discrete mesh where phi(s) is calculated
        n_basis: int, # number of basis vectors
        energy: float, # MeV, c.m.
        l: int # orbital angular momentum
    ):
        self.interaction = interaction
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
    

    def inhomogeneous_term(self, functional_operator, judges):
        '''
        Depending on the basis, the inhomogeneous term of our Galerkin equation
        changes, so we will again leave this up to the subclass to implement.
        '''
        raise NotImplementedError


class StandardBasis(Basis):
    def __init__(self,
        interaction: Interaction,
        theta_train: npt.ArrayLike, # training space
        s_mesh: npt.ArrayLike, # s = kr; discrete mesh where phi(s) is calculated
        n_basis: int, # number of basis vectors
        energy: float, # MeV, c.m.
        l: int, # orbital angular momentum
        use_svd: bool # use principal components?
    ):
        super().__init__(interaction, theta_train, s_mesh, n_basis, energy, l)

        schrodeq = SchroedingerEquation(self.interaction)
        self.all_vectors = np.array([
            schrodeq.phi(energy, theta, self.s_mesh, l) for theta in theta_train
        ]).T

        if use_svd:
            U, _, _ = np.linalg.svd(self.all_vectors, full_matrices=False)
            self.all_vectors = np.copy(U)
        
        self.vectors = np.copy(self.all_vectors[:, :self.n_basis])


    def phi_hat(self, coefficients):
        '''
        Given the coefficients, presumably from the solution of the Galerkin
        equation, return hat{phi}.
        '''
        return np.sum(coefficients * self.vectors, axis=1)
    
        
    def inhomogeneous_term(self, functional_operator, judges):
        '''
        Based on this basis, what should be used as the inhomogeneous term in
        the Galerkin equation?

        I don't think this is right!!! But I don't think I care because this was
        an old way of doing things that I am only attempting to keep around for
        legacy reasons. Note that I'm completely ignoring the functional
        operator and the judges.
        '''
        return self.s_mesh[0]*np.ones(self.n_basis)



class RelativeBasis(Basis):
    def __init__(self,
        interaction: Interaction,
        theta_train: npt.ArrayLike, # training space
        s_mesh: npt.ArrayLike, # s = kr; discrete mesh where phi(s) is calculated
        n_basis: int, # number of basis vectors
        energy: float, # MeV, c.m.
        l: int, # orbital angular momentum
        use_svd: bool # use principal components?
    ):
        super().__init__(interaction, theta_train, s_mesh, n_basis, energy, l)

        self.phi_0 = phi_free(self.s_mesh, l)

        schrodeq = SchroedingerEquation(self.interaction)
        self.all_vectors = np.array([
            schrodeq.phi(energy, theta, self.s_mesh, l) - self.phi_0 for theta in theta_train
        ]).T

        if use_svd:
            U, _, _ = np.linalg.svd(self.all_vectors, full_matrices=False)
            self.all_vectors = np.copy(U)
        
        self.vectors = np.copy(self.all_vectors[:, :self.n_basis])
    

    def phi_hat(self, coefficients):
        return self.phi_0 + np.sum(coefficients * self.vectors, axis=1)
    

    def inhomogeneous_term(self, functional_operator, judges):
        '''
        functional_operator is a matrix
        This does not currently work. It's a little harder to represent all of F
        as a matrix that will @ed with the basis than I thought.
        '''
        return -judges.T @ functional_operator @ self.phi_0


class CustomBasis(Basis):
    def __init__(self,
        vectors: npt.ArrayLike, # pre-computed solutions
        s_mesh: npt.ArrayLike, # s = kr; discrete mesh where phi(s) is calculated
        n_basis: int, # number of basis vectors
        energy: float, # MeV, c.m.
        l: int, # orbital angular momentum
        use_svd: bool, # use principal components?
        inh_term_function: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    ):
        '''
        Because the user can specify *any* basis, there is an additional
        argument required for instantiation. This argument is somewhat more
        complicated. It is a function that takes the judges (matrix) and functional
        operator F (matrix) returns the equivalent of < psi | F (phi_0) > in
        RelativeBasis. Presumably, somewhere in this function the replacement
        for phi_0 comes in? I'm not sure if this is going to work.
        '''
        super().__init__(None, None, s_mesh, n_basis, energy, l)

        self.all_vectors = np.copy(vectors)
        self.inh_term_function = inh_term_function

        if use_svd:
            U, _, _ = np.linalg.svd(self.all_vectors, full_matrices=False)
            self.all_vectors = np.copy(U)
        
        self.vectors = np.copy(self.all_vectors[:, :self.n_basis])
    

    def phi_hat(self, coefficients):
        return np.sum(coefficients * self.vectors, axis=1)

    def inhomogeneous_term(self, functional_operator, judges):
        return self.inh_term_function(judges, functional_operator)
