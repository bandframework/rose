'''
Useful utility functions that I don't want to clutter up other modules with.
'''
import numpy as np
import numpy.typing as npt
from scipy.sparse import diags

def finite_difference_first_derivative(
    s_mesh: npt.NDArray
):
    '''
    Computes a finite difference matrix that (when applied) represent the first
    derivative.

    :param s_mesh: Array of s points used to generate the matrix.
    :return: Matrix that generates the first derivative.
    :rtype: np.array

    '''
    ds = s_mesh[1] - s_mesh[0]
    assert np.all(np.abs(s_mesh[1:] - s_mesh[:-1] - ds) < 1e-14), '''
Spacing must be consistent throughout the entire mesh.
    '''
    ns = s_mesh.size
    D1 = diags([1, -8, 8, -1], [-2, -1, 1, 2], shape=(ns, ns)).toarray() / (12*ds)
    D1[0, 0] = -3/(2*ds)
    D1[0, 1] = 4/(2*ds)
    D1[0, 2] = -1/(2*ds)
    return D1


def finite_difference_second_derivative(
    s_mesh: npt.ArrayLike
):
    '''
    Computes a finite difference matrix that represents the second derivative
    (w.r.t. s or rho) operator in coordinate space.
    '''
    ds = s_mesh[1] - s_mesh[0]
    assert np.all(np.abs(s_mesh[1:] - s_mesh[:-1] - ds) < 1e-14), '''
Spacing must be consistent throughout the entire mesh.
    '''
    ns = s_mesh.size
    D2 = diags([-30, 16, 16, -1, -1], [0, 1, -1, 2, -2], shape=(ns, ns)).toarray() / (12*ds**2)
    D2[0, 0] = -2/ds**2
    D2[0, 1] = 1/ds**2
    D2[0, 2] = 0
    return D2
