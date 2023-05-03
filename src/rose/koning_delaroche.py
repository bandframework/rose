import numpy as np

from .interaction_eim import InteractionEIM
from .constants import DEFAULT_RHO_MESH

def Vv(E, v1, v2, v3, v4, Ef):
    '''
    Energy-dependent, volume-central strength
    - real term
    '''
    return v1 * (1 - v2*(E-Ef) + v3*(E-Ef)**2 - v4*(E-Ef)**3)


def Wv(E, w1, w2, Ef):
    '''
    Energy-dependent, volume-central strength
    - imaginary term
    '''
    return w1 * (E-Ef)**2 / ((E-Ef)**2 + w2**2)


def Wd(E, d1, d2, d3, Ef):
    '''
    Energy-dependent, surface-central strength
    - imaginary term (no real term)
    '''
    return d1 * (E-Ef)**2 / ((E-Ef)**2 + d3**2) * np.exp(-d2*(E-Ef))


def f_WS(r, R, a):
    '''
    Woods-Saxon potential
    '''
    return 1/(1 + np.exp((r-R)/a))


def fp_WS(r, R, a):
    '''
    Derivative of the Woods-Saxon potential
    '''
    return -1/a * np.exp((r-R)/a) / (1 + np.exp((r-R)/a))**2


def KD(r, E, v1, v2, v3, v4, w1, w2, d1, d2, d3, Ef, Rv, av, Rd, ad):
    '''
    Koning-Delaroche without the spin-orbit terms
    - Se Eq. (1) Nuclear Physics A 713 (2003) 231–310
    '''
    return -Vv(E, v1, v2, v3, v4, Ef) * f_WS(r, Rv, av) - \
           1j*Wv(E, w1, w2, Ef) * f_WS(r, Rv, av) - \
           1j * (-4*ad) * Wd(E, d1, d2, d3, Ef) * fp_WS(r, Rd, ad)


def KD_simple(r, alpha):
    vv, wv, wd, Rv, av, Rd, ad = alpha
    return vv * f_WS(r, Rv, av) + 1j*wv*f_WS(r, Rv, av) + \
           1j*(-4*ad)*wd * fp_WS(r, Rd, ad)


class KoningDelaroche(InteractionEIM):
    def __init__(self,
        mu: float,
        training_info: np.array,
        n_basis: int = 8,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None
    ):
        super().__init__(
            KD_simple, 7, mu, training_info=training_info, Z_1=0, Z_2=0,
            is_complex=True, n_basis=n_basis,
            explicit_training=explicit_training, n_train=n_train,
            rho_mesh=rho_mesh, match_points=match_points
        )