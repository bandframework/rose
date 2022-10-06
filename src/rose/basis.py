import numpy as np 
import numpy.typing as npt

class Basis:
    def __init__(self,
        phi_train: npt.ArrayLike,
        s_mesh: npt.ArrayLike,
    ):
        self.phi_train = np.copy(phi_train)
        self.d2_train = np.gradient(np.gradient(phi_train, s_mesh, axis=0), s_mesh, axis=0)

        U, _, _ = np.linalg.svd(phi_train, full_matrices=False)
        self.phi_svd = np.copy(U)
        self.d2_svd = np.gradient(np.gradient(self.phi_svd, s_mesh, axis=0), s_mesh, axis=0)
    

    def vectors(self,
        use_svd: bool = True, # use principal components (PCs) as the basis
        n_basis: int = 4 # How many PCs? If PCs are not used, all training vectors are used.
    ):
        if use_svd:
            return self.phi_svd[:, :n_basis]
        else:
            return self.phi_train