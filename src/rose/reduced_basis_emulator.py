import numpy as np

from .interaction import Interaction
from .schroedinger import SchroedingerEquation
from .basis import Basis
from .constants import HBARC
import numpy.typing as npt

DEFAULT_R_0 = 50.0 # fm
S_MIN = 1e-6
S_MAX = 50.0
NS = 2000

class ReducedBasisEmulator:
    def __init__(self,
        interaction: Interaction, # desired local interaction
        theta_train: npt.ArrayLike, # training points in parameter space
        energy: float, # center-of-mass energy (MeV)
        l: int, # angular momentum
        s_mesh: npt.ArrayLike = None,
        s_0: float = None,
        **kwargs # passed to SchroedingerEquation.solve_se
    ):
        self.energy = energy
        self.l = l
        self.se = SchroedingerEquation(interaction)

        if s_mesh is None:
            self.s_mesh = np.linspace(S_MIN, S_MAX, NS)
        else:
            self.s_mesh = np.copy(s_mesh)

        if s_0 is None:
            s_0 = np.sqrt(2*interaction.mu*energy/HBARC) * DEFAULT_R_0

        self.basis = Basis(
            np.array([
                self.se.true_phi_solver(self.energy, theta, self.s_mesh, self.l, s_0, **kwargs) for theta in theta_train
            ]),
            self.s_mesh
        )
    

    def emulate(self,
        theta: npt.ArrayLike,
        use_svd: bool = True,
        n_basis: int = 4
    ):
        utilde = self.se.interaction.tilde(self.s_mesh, theta, self.energy)[:, np.newaxis]
        phi_basis = self.basis.vectors(use_svd=use_svd, n_basis=n_basis)
        d2 = np.copy(self.basis.d2_svd if use_svd else self.basis.d2_train)

        A_right = -d2 + utilde * phi_basis - phi_basis
        A = phi_basis.T @ A_right
        A += np.vstack((
            phi_basis[0, :],
            phi_basis[0, :],
            phi_basis[0, :],
            phi_basis[0, :]
        ))
        b = self.s_mesh[0]*np.ones(4)
        x = np.linalg.solve(A, b)
        return np.sum(x * phi_basis, axis=1)