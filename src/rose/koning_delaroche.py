'''The Koning-Delaroche potential is a common optical potential for nuclear
scattering. It is provided here in simplified form specifically to address this
need.

See the [Koning-Delaroche
paper](https://www.sciencedirect.com/science/article/pii/S0375947402013210) for
details. Equation references are with respect to (w.r.t.) this paper.
'''

import numpy as np

from .interaction_eim import InteractionEIM
from .energized_interaction_eim import EnergizedInteractionEIM
from .constants import DEFAULT_RHO_MESH, MASS_PION

MAX_ARG = np.log(1/1e-16)
NUM_PARAMS = 15

def Vv(E, v1, v2, v3, v4, Ef):
    r'''energy-dependent, volume-central strength - real term, Eq. (7)
    '''
    return v1 * (1 - v2*(E-Ef) + v3*(E-Ef)**2 - v4*(E-Ef)**3)


def Wv(E, w1, w2, Ef):
    '''energy-dependent, volume-central strength - imaginary term, Eq. (7)
    '''
    return w1 * (E-Ef)**2 / ((E-Ef)**2 + w2**2)


def Wd(E, d1, d2, d3, Ef):
    '''energy-dependent, surface-central strength - imaginary term (no real
        term), Eq. (7)
    '''
    return d1 * (E-Ef)**2 / ((E-Ef)**2 + d3**2) * np.exp(-d2*(E-Ef))


def Vso(E, vso1, vso2, E_f):
    '''energy-dependent, spin-orbit strength --- real term, Eq. (7)'''
    return vso1 * np.exp(-vso2*(E-E_f))


def Wso(E, wso1, wso2, E_f):
    '''energy-dependent, spin-orbit strength --- imaginary term, Eq. (7)'''
    return wso1 * (E - E_f)**2 / ((E - E_f)**2 + wso2**2)


def woods_saxon(r, R, a):
    '''Woods-Saxon potential'''
    return 1/(1 + np.exp((r-R)/a))


def woods_saxon_safe(r, R, a):
    '''Woods-Saxon potential

     * avoids `exp` overflows

     '''
    x = (r-R)/a
    if isinstance(x, float):
        return 1/(1 + np.exp(x)) if x < MAX_ARG else 0
    else:
        ii = np.where(x <= MAX_ARG)[0]
        jj = np.where(x > MAX_ARG)[0]
        return np.hstack([1/(1 + np.exp(x[ii])), np.zeros(jj.size)])
    

def woods_saxon_prime(r, R, a):
    '''derivative of the Woods-Saxon potential w.r.t. $r$
    '''
    return -1/a * np.exp((r-R)/a) / (1 + np.exp((r-R)/a))**2


def woods_saxon_prime_safe(r, R, a):
    '''derivative of the Woods-Saxon potential w.r.t. $r$

    * avoids `exp` overflows

    '''
    x = (r-R)/a
    if isinstance(x, float):
        return -1/a * np.exp(x)/(1 + np.exp(x))**2 if x < MAX_ARG else 0
    else:
        ii = np.where(x <= MAX_ARG)[0]
        jj = np.where(x > MAX_ARG)[0]
        return np.hstack([-1/a * np.exp(x[ii])/(1 + np.exp(x[ii]))**2, np.zeros(jj.size)])


def KD(r, E, v1, v2, v3, v4, w1, w2, d1, d2, d3, Ef, Rv, av, Rd, ad):
    '''Koning-Delaroche without the spin-orbit terms - Eq. (1)'''
    return -Vv(E, v1, v2, v3, v4, Ef) * woods_saxon(r, Rv, av) - \
           1j*Wv(E, w1, w2, Ef) * woods_saxon(r, Rv, av) - \
           1j * (-4*ad) * Wd(E, d1, d2, d3, Ef) * woods_saxon_prime(r, Rd, ad)


def decompose_alpha(alpha):
    r'''Splits the parameter-space vector into non-spin-orbit and spin-orbit
    parameters.

    Parameters:
        alpha (ndarray): interaction parameters
    
    Returns:
        parameters (tuple): 2-tuple of non-spin-orbit (`parameters[0]`) and
            spin-orbit parameters (`parameters[1]`)

    '''
    vv, rv, av, wv, rwv, awv, wd, rd, ad, vso, rso, aso, wso, rwso, awso = alpha
    return (vv, rv, av, wv, rwv, awv, wd, rd, ad), (vso, rso, aso, wso, rwso, awso)

def KD_simple(r, alpha):
    r'''simplified Koning-Delaroche without the spin-orbit terms

    Take Eq. (1) and remove the energy dependence of the coefficients.
    '''
    vv, rv, av, wv, rwv, awv, wd, rd, ad = decompose_alpha(alpha)[0]
    return -vv * woods_saxon_safe(r, rv, av) - \
        1j*wv * woods_saxon_safe(r, rwv, awv) - \
        1j*(-4*ad)*wd * woods_saxon_prime_safe(r, rd, ad)


def KD_simple_so(r, alpha, lds):
    r'''simplified Koning-Delaroche *with* the spin-orbit terms

    Take Eq. (1) and remove the energy dependence of the coefficients.

    lds: l • s = 1/2 * (j(j+1) - l(l+1) - s(s+1))
    '''
    vso, rso, aso, wso, rwso, awso = decompose_alpha(alpha)[1]
    return lds * vso/MASS_PION**2/r * woods_saxon_prime_safe(r, rso, aso) + \
        1j*wso/MASS_PION**2/r*woods_saxon_prime_safe(r, rwso, awso)


class KoningDelaroche(InteractionEIM):
    r'''Koning-Delaroche potential (without energy-dependent strength
    coefficients) for arbitrary systems defined by `mu`, `energy`, `ell`, `Z_1`,
    and `Z_2`.
    '''
    def __init__(self,
        mu: float,
        ell: int,
        energy: float,
        training_info: np.array,
        n_basis: int = 8,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None
    ):
        r'''Wraps the Koning-Delaroche potential into a `rose`-friendly class.
        Saves system-specific information.
        
        Parameters:
            mu (float): reduced mass of the 2-body system
            ell (int): angular momentum
            energy (float): center-of-mass, scattering energy
            training_info (ndarray): either (1) parameters bounds or (2) explicit training points

                If (1):
                This is a 2-column matrix. The first column are the lower
                bounds. The second are the upper bounds. Each row maps to a
                single parameter.

                If (2):
                This is an MxN matrix. N is the number of parameters. M is the
                number of samples.
            n_basis (int): number of basis states to use for EIM
            explicit_training (bool): True implies training_info case (2); False implies (1)
            n_train (int): how many training samples to use
            rho_mesh (ndarray):  $\rho$ (or $s$) grid values
            match_points (ndarray): $\rho$ points at which we demand the EIMed
                potential match the true potential

        Returns:
            instance (KoningDelaroche): instance of the class

        ''' 
        super().__init__(
            KD_simple_so, NUM_PARAMS, mu, ell, energy, training_info=training_info, Z_1=0, Z_2=0,
            is_complex=True, n_basis=n_basis,
            explicit_training=explicit_training, n_train=n_train,
            rho_mesh=rho_mesh, match_points=match_points
        )


class EnergizedKoningDelaroche(EnergizedInteractionEIM):
    def __init__(self,
        mu: float,
        ell: int,
        training_info: np.array,
        n_basis: int = 8,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None
    ):
        r'''Wraps the Koning-Delaroche potential into a `rose`-friendly class.
        Saves system-specific information. Allows the user to emulate across
        energies.

        * **Does not (yet) support Coulomb.**
        
        Parameters:
            mu (float): reduced mass of the 2-body system
            ell (int): angular momentum
            training_info (ndarray): either (1) parameters bounds or (2) explicit training points

                If (1):
                This is a 2-column matrix. The first column are the lower
                bounds. The second are the upper bounds. Each row maps to a
                single parameter.

                If (2):
                This is an MxN matrix. N is the number of parameters. M is the
                number of samples.
            n_basis (int): number of basis states to use for EIM
            explicit_training (bool): True implies training_info case (2); False implies (1)
            n_train (int): how many training samples to use
            rho_mesh (ndarray):  $\rho$ (or $s$) grid values
            match_points (ndarray): $\rho$ points at which we demand the EIMed
                potential match the true potential
        
        Returns:
            instance (EnergizedKoningDelaroche): instance of the class
        ''' 
        super().__init__(
            KD_simple_so, NUM_PARAMS, mu, ell, training_info=training_info, Z_1=0, Z_2=0,
            is_complex=True, n_basis=n_basis,
            explicit_training=explicit_training, n_train=n_train,
            rho_mesh=rho_mesh, match_points=match_points
        )
