"""The Whitehead-Lim-Holt potential is a global mcroscopic optical potential for nuclear
scattering.

See the [paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.182502) for
details. Equation references are with respect to (w.r.t.) this paper.
"""
from pathlib import Path

import json
import numpy as np
from numba import njit

from .interaction_eim import InteractionEIMSpace
from .energized_interaction_eim import EnergizedInteractionEIMSpace
from .constants import DEFAULT_RHO_MESH, MASS_PION, HBARC, ALPHA
from .utility import (
    kinematics,
    Projectile,
    woods_saxon,
    woods_saxon_safe,
    woods_saxon_prime,
    woods_saxon_prime_safe,
)

MAX_ARG = np.log(1 / 1e-16)
NUM_PARAMS = 12


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
    uv, rv, av, uw, rw, aw, ud, rd, ad, uso, rso, aso = alpha
    return (uv, rv, av, uw, rw, aw, ud, rd, ad), (uso, rso, aso)


@njit
def WLH_so(r, alpha, lds):
    r"""simplified Koning-Delaroche spin-orbit terms

    Take Eq. (1) and remove the energy dependence of the coefficients.

    lds: l • s = 1/2 * (j(j+1) - l(l+1) - s(s+1))
    """
    uso, rso, aso = decompose_alpha(alpha)[1]
    return lds * (uso / MASS_PION**2) / r * woods_saxon_prime_safe(r, rso, aso)


@njit
def WLH(r, alpha):
    """WLH without the spin-orbit term - Eq. (2)."""

    uv, rv, av, uw, rw, aw, ud, rd, ad = decompose_alpha(alpha)[0]
    return (
        -uv * woods_saxon(r, rv, av)
        - 1j * uw * woods_saxon(r, rw, aw)
        - 1j * (-4 * ad) * ud * woods_saxon_prime(r, rd, ad)
    )


class EnergizedWLH(EnergizedInteractionEIMSpace):
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
        method="collocation",
    ):
        r"""Wraps the WLH potential into a `rose`-friendly class.
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
            instance (EnergizedWLH): instance of the class
        """
        n_params = NUM_PARAMS + 1  # include energy
        if mu is None:
            n_params = NUM_PARAMS + 2  # include mu and energy

        super().__init__(
            l_max=l_max,
            coordinate_space_potential=WLH,
            n_theta=n_params,
            mu=mu,
            training_info=training_info,
            Z_1=0,
            Z_2=0,
            is_complex=True,
            spin_orbit_term=WLH_so,
            n_basis=n_basis,
            explicit_training=explicit_training,
            n_train=n_train,
            rho_mesh=rho_mesh,
            match_points=match_points,
            method=method,
        )


class WLHGlobal:
    r"""Global optical potential in WLH form."""

    def __init__(self, projectile: Projectile, param_fpath: Path = None):
        r"""
        Parameters:
            projectile : neutron or proton?
            param_fpath : path to json file encoding parameter values.
                Defaults to data/WLH_mean.json
        """
        if param_fpath is None:
            param_fpath = Path(__file__).parent.resolve() / Path(
                "../../data/WLH_mean.json"
            )

        if projectile == Projectile.neutron:
            tag = "_n"
        elif projectile == Projectile.proton:
            tag = "_p"
        else:
            raise RuntimeError(
                "WLHGlobal is defined only for neutron and proton projectiles"
            )

        self.projectile = projectile

        self.param_fpath = param_fpath
        with open(self.param_fpath) as f:
            data = json.load(f)
            self.uv0 = data["WLHReal_V0" + tag]
            self.uv1 = data["WLHReal_V1" + tag]
            self.uv2 = data["WLHReal_V2" + tag]
            self.uv3 = data["WLHReal_V3" + tag]
            self.uv4 = data["WLHReal_V4" + tag]
            self.uv5 = data["WLHReal_V5" + tag]
            self.uv6 = data["WLHReal_V6" + tag]
            self.rv0 = data["WLHReal_r0" + tag]
            self.rv1 = data["WLHReal_r1" + tag]
            self.rv2 = data["WLHReal_r2" + tag]
            self.rv3 = data["WLHReal_r3" + tag]
            self.av0 = data["WLHReal_a0" + tag]
            self.av1 = data["WLHReal_a1" + tag]
            self.av2 = data["WLHReal_a2" + tag]
            self.av3 = data["WLHReal_a3" + tag]
            self.av4 = data["WLHReal_a4" + tag]
            self.uw0 = data["WLHImagVolume_W0" + tag]
            self.uw1 = data["WLHImagVolume_W1" + tag]
            self.uw2 = data["WLHImagVolume_W2" + tag]
            self.uw3 = data["WLHImagVolume_W3" + tag]
            self.uw4 = data["WLHImagVolume_W4" + tag]
            self.rw0 = data["WLHImagVolume_r0" + tag]
            self.rw1 = data["WLHImagVolume_r1" + tag]
            self.rw2 = data["WLHImagVolume_r2" + tag]
            self.rw3 = data["WLHImagVolume_r3" + tag]
            self.rw4 = data["WLHImagVolume_r4" + tag]
            self.rw5 = data["WLHImagVolume_r5" + tag]
            self.aw0 = data["WLHImagVolume_a0" + tag]
            self.aw1 = data["WLHImagVolume_a1" + tag]
            self.aw2 = data["WLHImagVolume_a2" + tag]
            self.aw3 = data["WLHImagVolume_a3" + tag]
            self.aw4 = data["WLHImagVolume_a4" + tag]
            self.ud0 = data["WLHImagSurface_W0" + tag]
            self.ud1 = data["WLHImagSurface_W1" + tag]
            self.ud3 = data["WLHImagSurface_W2" + tag]
            self.ud4 = data["WLHImagSurface_W3" + tag]
            self.rd0 = data["WLHImagSurface_r0" + tag]
            self.rd1 = data["WLHImagSurface_r1" + tag]
            self.rd2 = data["WLHImagSurface_r2" + tag]
            self.ad0 = data["WLHImagSurface_a0" + tag]
            self.uso0 = data["WLHRealSpinOrbit_V0" + tag]
            self.uso1 = data["WLHRealSpinOrbit_V1" + tag]
            self.rso0 = data["WLHRealSpinOrbit_r0" + tag]
            self.rso1 = data["WLHRealSpinOrbit_r1" + tag]
            self.aso0 = data["WLHRealSpinOrbit_a0" + tag]
            self.aso1 = data["WLHRealSpinOrbit_a1" + tag]

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
        factor = 1.0
        if self.projectile == Projectile.neutron:
            factor *= -1.0

        uv = (
            self.uv0
            - self.uv1 * E_com
            + self.uv2 * E_com**2
            + self.uv3 * E_com**3
            + factor * (self.uv4 - self.uv5 * E_com + self.uv6 * E_com**2) * delta
        )
        rv = (
            self.rv0
            - self.rv1 * E_com
            + self.rv2 * E_com**2
            - self.rv3 * A ** (-1.0 / 3)
        )
        av = (
            self.av0
            - factor * self.av1 * E_com
            - self.av2 * E_com**2
            - (self.av3 - self.av4 * delta) * delta
        )

        uw = (
            self.uw0
            + self.uw1 * E_com
            - self.uw2 * E_com**2
            + (factor * self.uw3 - self.uw4 * E_com) * delta
        )
        rw = (
            self.rw0
            + (self.rw1 + self.rw2 * A) / (self.rw3 + A + self.rw4 * E_com)
            + self.rw5 * E_com**2
        )
        aw = (
            self.aw0
            - (self.aw1 * E_com) / (-self.aw2 - E_com)
            + (self.aw3 - self.aw4 * E_com) * delta
        )

        ud = self.ud0 - self.ud1 * E_com - (self.ud3 - self.ud4 * E_com) * delta
        rd = self.rd0 - self.rd1 * E_com - self.rd2 * A ** (-1.0 / 3)
        ad = self.ad0

        uso = self.uso0 - self.uso1 * A
        rso = self.rso0 - self.rso1 * A ** (-1.0 / 3.0)
        aso = self.aso0 - self.aso1 * A

        params = np.array(
            [
                uv,
                rv * A ** (1.0 / 3.0),
                av,
                uw,
                rw * A ** (1.0 / 3.0),
                aw,
                ud,
                rd * A ** (1.0 / 3.0),
                ad,
                uso,
                rso * A ** (1.0 / 3.0),
                aso,
            ]
        )
        R_C = rv
        return (
            (mu, E_com, k, eta, R_C),
            params,
        )
