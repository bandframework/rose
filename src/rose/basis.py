import numpy as np 
from scipy.sparse import diags
import numpy.typing as npt

def finite_difference_second_derivative(
    s_mesh: npt.ArrayLike
):
    ds = s_mesh[1] - s_mesh[0]
    assert np.all(s_mesh[1:] - s_mesh[:-1] == ds), '''
Spacing must be consistent throughout the entire mesh.
    '''

    ns = s_mesh.size
    D2 = diags([-30, 16, 16, -1, -1], [0, 1, -1, 2, -2], shape=(ns, ns)).toarray() / (12*ds**2)
    D2[0, 0] = -2/ds**2
    D2[0, 1] = 1/ds**2
    D2[0, 2] = 0
    return D2

class Basis:
    def __init__(self,
        phi_train: npt.ArrayLike,
        s_mesh: npt.ArrayLike,
    ):
        self.phi_train = np.copy(phi_train)
        self.d2_operator = finite_difference_second_derivative(s_mesh)

        self.d2_train = self.d2_operator @ self.phi_train

        U, _, _ = np.linalg.svd(phi_train, full_matrices=False)
        self.phi_svd = np.copy(U)
        self.d2_svd = self.d2_operator @ self.phi_svd
    

    def vectors(self,
        use_svd: bool = True, # use principal components (PCs) as the basis
        n_basis: int = 4 # How many PCs? If PCs are not used, all training vectors are used.
    ):
        if use_svd:
            return self.phi_svd[:, :n_basis]
        else:
            return self.phi_train