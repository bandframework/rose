"""The Koning-Delaroche potential is a common optical potential for nuclear
scattering. It is provided here in simplified form specifically to address this
need.

See the [Koning-Delaroche
paper](https://www.sciencedirect.com/science/article/pii/S0375947402013210) for
details. Equation references are with respect to (w.r.t.) this paper.
"""
from pathlib import Path

import json
import numpy as np
from numba import njit

from .interaction_eim import InteractionEIMSpace
from .energized_interaction_eim import EnergizedInteractionEIMSpace
from .constants import DEFAULT_RHO_MESH, MASS_PION, HBARC, ALPHA
from .utility import kinematics, Projectile

MAX_ARG = np.log(1 / 1e-16)
NUM_PARAMS = 15


@njit
def Vv(E, v1, v2, v3, v4, Ef):
    r"""energy-dependent, volume-central strength - real term, Eq. (7)"""
    return v1 * (1 - v2 * (E - Ef) + v3 * (E - Ef) ** 2 - v4 * (E - Ef) ** 3)


@njit
def Wv(E, w1, w2, Ef):
    """energy-dependent, volume-central strength - imaginary term, Eq. (7)"""
    return w1 * (E - Ef) ** 2 / ((E - Ef) ** 2 + w2**2)


@njit
def Wd(E, d1, d2, d3, Ef):
    """energy-dependent, surface-central strength - imaginary term (no real
    term), Eq. (7)
    """
    return d1 * (E - Ef) ** 2 / ((E - Ef) ** 2 + d3**2) * np.exp(-d2 * (E - Ef))


@njit
def Vso(E, vso1, vso2, E_f):
    """energy-dependent, spin-orbit strength --- real term, Eq. (7)"""
    return vso1 * np.exp(-vso2 * (E - E_f))


@njit
def Wso(E, wso1, wso2, E_f):
    """energy-dependent, spin-orbit strength --- imaginary term, Eq. (7)"""
    return wso1 * (E - E_f) ** 2 / ((E - E_f) ** 2 + wso2**2)


@njit
def woods_saxon(r, R, a):
    """Woods-Saxon potential"""
    return 1 / (1 + np.exp((r - R) / a))


@njit
def woods_saxon_safe(r, R, a):
    """Woods-Saxon potential

    * avoids `exp` overflows

    """
    x = (r - R) / a
    if isinstance(x, float):
        return 1 / (1 + np.exp(x)) if x < MAX_ARG else 0
    else:
        ii = np.where(x <= MAX_ARG)[0]
        jj = np.where(x > MAX_ARG)[0]
        return np.hstack((1 / (1 + np.exp(x[ii])), np.zeros(jj.size)))


@njit
def woods_saxon_prime(r, R, a):
    """derivative of the Woods-Saxon potential w.r.t. $r$"""
    return -1 / a * np.exp((r - R) / a) / (1 + np.exp((r - R) / a)) ** 2


@njit
def woods_saxon_prime_safe(r, R, a):
    """derivative of the Woods-Saxon potential w.r.t. $r$

    * avoids `exp` overflows

    """
    x = (r - R) / a
    if isinstance(x, float):
        return -1 / a * np.exp(x) / (1 + np.exp(x)) ** 2 if x < MAX_ARG else 0
    else:
        ii = np.where(x <= MAX_ARG)[0]
        jj = np.where(x > MAX_ARG)[0]
        return np.hstack(
            (-1 / a * np.exp(x[ii]) / (1 + np.exp(x[ii])) ** 2, np.zeros(jj.size))
        )


@njit
def KD(r, E, v1, v2, v3, v4, w1, w2, d1, d2, d3, Ef, Rv, av, Rd, ad):
    """Koning-Delaroche without the spin-orbit terms - Eq. (1)"""
    return (
        -Vv(E, v1, v2, v3, v4, Ef) * woods_saxon(r, Rv, av)
        - 1j * Wv(E, w1, w2, Ef) * woods_saxon(r, Rv, av)
        - 1j * (-4 * ad) * Wd(E, d1, d2, d3, Ef) * woods_saxon_prime(r, Rd, ad)
    )


@njit
def decompose_alpha(alpha):
    r"""Splits the parameter-space vector into non-spin-orbit and spin-orbit
    parameters.

    Parameters:
        alpha (ndarray): interaction parameters

    Returns:
        parameters (tuple): 2-tuple of non-spin-orbit (`parameters[0]`) and
            spin-orbit parameters (`parameters[1]`)

    """
    vv, rv, av, wv, rwv, awv, wd, rd, ad, vso, rso, aso, wso, rwso, awso = alpha
    return (vv, rv, av, wv, rwv, awv, wd, rd, ad), (vso, rso, aso, wso, rwso, awso)


@njit
def KD_simple(r, alpha):
    r"""simplified Koning-Delaroche without the spin-orbit terms

    Take Eq. (1) and remove the energy dependence of the coefficients.
    """
    vv, rv, av, wv, rwv, awv, wd, rd, ad = decompose_alpha(alpha)[0]
    return (
        -vv * woods_saxon_safe(r, rv, av)
        - 1j * wv * woods_saxon_safe(r, rwv, awv)
        - 1j * (-4 * ad) * wd * woods_saxon_prime_safe(r, rd, ad)
    )


@njit
def KD_simple_so(r, alpha, lds):
    r"""simplified Koning-Delaroche spin-orbit terms

    Take Eq. (1) and remove the energy dependence of the coefficients.

    lds: l • s = 1/2 * (j(j+1) - l(l+1) - s(s+1))
    """
    vso, rso, aso, wso, rwso, awso = decompose_alpha(alpha)[1]
    return lds * vso / MASS_PION**2 / r * woods_saxon_prime_safe(
        r, rso, aso
    ) + 1j * wso / MASS_PION**2 / r * woods_saxon_prime_safe(r, rwso, awso)


class KoningDelaroche(InteractionEIMSpace):
    r"""Koning-Delaroche potential (without energy-dependent strength
    coefficients) for arbitrary systems defined by `mu`, `energy`, `ell`, `Z_1`,
    and `Z_2`.
    """

    def __init__(
        self,
        energy: float,
        training_info: np.array,
        mu: float,
        l_max=20,
        n_basis: int = 8,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None,
    ):
        r"""Wraps the Koning-Delaroche potential into a `rose`-friendly class.
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

        """
        super().__init__(
            KD_simple,
            NUM_PARAMS,
            mu,
            energy,
            training_info=training_info,
            Z_1=0,
            Z_2=0,
            l_max=l_max,
            is_complex=True,
            spin_orbit_potential=KD_simple_so,
            n_basis=n_basis,
            explicit_training=explicit_training,
            n_train=n_train,
            rho_mesh=rho_mesh,
            match_points=match_points,
        )


class EnergizedKoningDelaroche(EnergizedInteractionEIMSpace):
    def __init__(
        self,
        training_info: np.array,
        mu: float = None,
        l_max=20,
        n_basis: int = 8,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None,
    ):
        r"""Wraps the Koning-Delaroche potential into a `rose`-friendly class.
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
        """
        n_params = NUM_PARAMS + 1  # include energy
        if mu is None:
            n_params = 17  # include mu
        super().__init__(
            KD_simple,
            n_params,
            mu,
            training_info=training_info,
            Z_1=0,
            Z_2=0,
            l_max=l_max,
            is_complex=True,
            spin_orbit_potential=KD_simple_so,
            n_basis=n_basis,
            explicit_training=explicit_training,
            n_train=n_train,
            rho_mesh=rho_mesh,
            match_points=match_points,
        )


class KDGlobal:
    r"""Global optical potential in Koning-Delaroche form."""

    def __init__(self, projectile: Projectile, param_fpath: Path = None):
        r"""
        Parameters:
            projectile : neutron or proton?
            param_fpath : path to json file encoding parameter values. Defaults to data/KD_default.json
        """
        if param_fpath is None:
            param_fpath = Path(__file__).parent.resolve() / Path(
                "../../data/KD_default.json"
            )

        if projectile == Projectile.neutron:
            tag = "_n"
        elif projectile == Projectile.proton:
            tag = "_p"
        else:
            raise RuntimeError(
                "KDGlobal is defined only for neutron and proton projectiles"
            )

        self.projectile = projectile

        self.param_fpath = param_fpath
        with open(self.param_fpath) as f:
            data = json.load(f)

            # real central depth
            self.v1_0 = data["KDHartreeFock_V1_0"]
            self.v1_asymm = data["KDHartreeFock_V1_asymm"]
            self.v1_A = data["KDHartreeFock_V1_A"]
            self.v2_0 = data["KDHartreeFock_V2_0" + tag]
            self.v2_A = data["KDHartreeFock_V2_A" + tag]
            self.v3_0 = data["KDHartreeFock_V3_0" + tag]
            self.v3_A = data["KDHartreeFock_V3_A" + tag]
            self.v4_0 = data["KDHartreeFock_V4_0"]

            # real central form
            self.rv_0 = data["KDHartreeFock_r_0"]
            self.rv_A = data["KDHartreeFock_r_A"]
            self.av_0 = data["KDHartreeFock_a_0"]
            self.av_A = data["KDHartreeFock_a_A"]

            # imag volume depth
            self.w1_0 = data["KDImagVolume_W1_0" + tag]
            self.w1_A = data["KDImagVolume_W1_A" + tag]
            self.w2_0 = data["KDImagVolume_W2_0"]
            self.w2_A = data["KDImagVolume_W2_A"]

            # imag surface depth
            self.d1_0 = data["KDImagSurface_D1_0"]
            self.d1_asymm = data["KDImagSurface_D1_asymm"]
            self.d2_0 = data["KDImagSurface_D2_0"]
            self.d2_A = data["KDImagSurface_D2_A"]
            self.d2_A2 = data["KDImagSurface_D2_A2"]
            self.d2_A3 = data["KDImagSurface_D2_A3"]
            self.d3_0 = data["KDImagSurface_D3_0"]

            # imag surface form
            self.rd_0 = data["KDImagSurface_r_0"]
            self.rd_A = data["KDImagSurface_r_A"]
            self.ad_0 = data["KDImagSurface_a_0" + tag]
            self.ad_A = data["KDImagSurface_a_A" + tag]

            # real spin orbit depth
            self.Vso1_0 = data["KDRealSpinOrbit_V1_0"]
            self.Vso1_A = data["KDRealSpinOrbit_V1_A"]
            self.Vso2_0 = data["KDRealSpinOrbit_V2_0"]

            # imag spin orbit form
            self.Wso1_0 = data["KDImagSpinOrbit_W1_0"]
            self.Wso2_0 = data["KDImagSpinOrbit_W2_0"]

            # spin orbit form
            self.rso_0 = data["KDRealSpinOrbit_r_0"]
            self.rso_A = data["KDRealSpinOrbit_r_A"]
            self.aso_0 = data["KDRealSpinOrbit_a_0"]

            # fermi energy
            if self.projectile == Projectile.neutron:
                self.Ef_0 = 11.2814
                self.Ef_A = 0.02646
            else:
                self.Ef_0 = -8.4075
                self.Ef_A = 1.01378

            # Coulomb
            if self.projectile == Projectile.proton:
                self.rc_0 = data["KDCoulomb_r_C_0"]
                self.rc_A = data["KDCoulomb_r_C_A"]
                self.rc_A2 = data["KDCoulomb_r_C_A2"]

    def get_params(self, A, Z, E_lab=None, E_com=None):
        """
        Calculates Koning-Delaroche global neutron-nucleus OMP parameters for given A, Z,
        and COM-frame energy, returns params in form useable by EnergizedKoningDelaroche
        """

        if self.projectile == Projectile.neutron:
            projectile = (1, 0)
        elif self.projectile == Projectile.proton:
            projectile = (1, 1)

        mu, E_com, k = kinematics((A, Z), projectile, E_lab=E_lab, E_com=E_com)
        eta = 0
        if self.projectile == Projectile.proton:
            k_c = ALPHA * Z * mu
            eta = k_c / k

        N = A - Z
        delta = (N - Z) / A
        factor = 1
        if self.projectile == Projectile.proton:
            delta *= -1.0
            factor *= -1

        # fermi energy
        Ef = self.Ef_0 + self.Ef_A * A

        # real central depth
        v1 = self.v1_0 - self.v1_asymm * delta - self.v1_A * A
        v2 = self.v2_0 - self.v2_A * A * factor
        v3 = self.v3_0 - self.v3_A * A * factor
        v4 = self.v4_0
        vv = Vv(E_com, v1, v2, v3, v4, Ef)

        # real central form
        rv = self.rv_0 - self.rv_A * A ** (-1.0 / 3.0)
        av = self.av_0 - self.av_A * A

        # imag volume depth
        w1 = self.w1_0 + self.w1_A * A
        w2 = self.w2_0 + self.w2_A * A
        wv = Wv(E_com, w1, w2, Ef)

        # imag volume form
        rwv = rv
        awv = av

        # imag surface depth
        d1 = self.d1_0 + self.d1_asymm * delta
        d2 = self.d2_0 + self.d2_A / (1 + np.exp((A - self.d2_A3) / self.d2_A2))
        d3 = self.d3_0
        wd = Wd(E_com, d1, d2, d3, Ef)

        # imag surface form
        rd = self.rd_0 - self.rd_A * A ** (1.0 / 3.0)
        ad = self.ad_0 - self.ad_A * A * factor

        # real spin orbit depth
        vso1 = self.Vso1_0 + self.Vso1_A * A
        vso2 = self.Vso2_0
        vso = Vso(E_com, vso1, vso2, Ef)

        # real spin orbit form
        rso = self.rso_0 - self.rso_A * A ** (-1.0 / 3.0)
        aso = self.aso_0

        # imag spin orbit form
        wso1 = self.Wso1_0
        wso2 = self.Wso2_0
        wso = Wso(E_com, wso1, wso2, Ef)

        # imag spin orbit form
        rwso = rso
        awso = aso

        # Coulomb radius
        R_C = 0
        if self.projectile == Projectile.proton:
            rc0 = (
                self.rc_0
                + self.rc_A * A ** (-2.0 / 3.0)
                + self.rc_A2 * A ** (-5.0 / 3.0)
            )
            R_C = rc0 * A ** (1.0 / 3.0)

        # 15 params total
        params = np.array(
            [
                vv,
                rv * A ** (1.0 / 3.0),
                av,
                wv,
                rwv * A ** (1.0 / 3.0),
                awv,
                wd,
                rd * A ** (1.0 / 3.0),
                ad,
                vso,
                rso * A ** (1.0 / 3.0),
                aso,
                wso,
                rwso * A ** (1.0 / 3.0),
                awso,
            ]
        )

        return (
            (mu, E_com, k, eta, R_C),
            params,
        )
