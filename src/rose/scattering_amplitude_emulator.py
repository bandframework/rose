import pickle
import numpy as np
from scipy.special import eval_legendre, gamma
from tqdm import tqdm
from numba import njit

from .interaction import InteractionSpace
from .reduced_basis_emulator import ReducedBasisEmulator
from .constants import DEFAULT_RHO_MESH, DEFAULT_ANGLE_MESH
from .schroedinger import SchroedingerEquation
from .basis import RelativeBasis, CustomBasis, Basis
from .utility import (
    eval_assoc_legendre,
    xs_calc_neutral,
    xs_calc_coulomb,
    NucleonNucleusXS,
)


class ScatteringAmplitudeEmulator:
    @classmethod
    def load(obj, filename):
        r"""Loads a previously trained emulator.

        Parameters:
            filename (string): name of file

        Returns:
            emulator (ScatteringAmplitudeEmulator): previously trainined `ScatteringAmplitudeEmulator`

        """
        with open(filename, "rb") as f:
            sae = pickle.load(f)
        return sae

    @classmethod
    def HIFI_solver(
        cls,
        interaction_space: InteractionSpace,
        base_solver: SchroedingerEquation = None,
        l_max: int = None,
        angles: np.array = DEFAULT_ANGLE_MESH,
        verbose: bool = True,
        Smatrix_abs_tol: float = 1.0e-6,
        s_mesh=None,
    ):
        r"""Sets up a ScatteringAmplitudeEmulator without any emulation capabilities, for use purely
            as a high-fidelity solver, for which the exact_* functions will be used.

        Parameters:
            interaction_space (InteractionSpace): local interaction up to (and including $\ell_\max$)
            base_solver : the solver used. Must be an instance of SchroedingerEquation or a derived
                class of it. The solvers for each `interaction` in `interaction_space` will be
                constructed using `base_solver.clone_for_new_interaction`. Defaults to the base class
                using Runge-Kutta; `SchroedingerEquation`
            l_max (int): maximum angular momentum to include in the sum approximating the cross section
            angles (ndarray): Differential cross sections are functions of the
                angles. These are the specific values at which the user wants to
                emulate the cross section.
            s_0 (float): $s$ point where the phase shift is extracted
            verbose (bool): Do you want the class to print out warnings?
            Smatrix_abs_tol : absolute tolerance for deviation of real part of S-matrix amplitudes
                from 1, used as criteria to stop calculation ig higher partial waves are negligble
            s_mesh (ndarray): $s$ (or $\rho$) grid on which wave functions are evaluated

        Returns:
            sae (ScatteringAmplitudeEmulator): scattering amplitude emulator

        """
        if base_solver is None:
            base_solver = SchroedingerEquation.make_base_solver()

        if s_mesh is None:
            s_mesh = np.linspace(
                base_solver.domain[0],
                base_solver.domain[1],
                SchroedingerEquation.DEFAULT_NUM_PTS,
            )

        assert (
            s_mesh[0] >= base_solver.domain[0] and s_mesh[-1] <= base_solver.domain[1]
        )

        bases = []
        for interaction_list in interaction_space.interactions:
            basis_list = []
            for interaction in interaction_list:
                solver = base_solver.clone_for_new_interaction(interaction)
                basis = Basis(solver, None, s_mesh, None)
                basis_list.append(basis)
            bases.append(basis_list)

        return cls(
            interaction_space,
            bases,
            l_max,
            angles,
            s_0=base_solver.s_0,
            verbose=verbose,
            Smatrix_abs_tol=Smatrix_abs_tol,
            initialize_emulator=False,
        )

    @classmethod
    def from_train(
        cls,
        interaction_space: InteractionSpace,
        alpha_train: np.array,
        base_solver: SchroedingerEquation = None,
        l_max: int = None,
        angles: np.array = DEFAULT_ANGLE_MESH,
        n_basis: int = 4,
        use_svd: bool = True,
        Smatrix_abs_tol: float = 1.0e-6,
        s_mesh: np.array = None,
        **basis_kwargs,
    ):
        r"""Trains a reduced-basis emulator based on the provided interaction and training space.

        Parameters:
            interaction_space (InteractionSpace): local interaction up to (and including $\ell_\max$)
            alpha_train (ndarray): training points in parameter space; shape = (n_points, n_parameters)
            base_solver : the solver used for training the emulator, and for calculations of exact
                observables. Must be an instance of SchroedingerEquation or a derived class of it.
                The solvers for each `interaction` in `interaction_space` will be constructed using
            l_max (int): maximum angular momentum to include in the sum approximating the cross section
            angles (ndarray): Differential cross sections are functions of the
                angles. These are the specific values at which the user wants to
                emulate the cross section.
            n_basis (int): number of basis vectors for $\hat{\phi}$ expansion
            use_svd (bool): Use principal components of training wave functions?
            s_mesh (ndarray): $s$ (or $\rho$) grid on which wave functions are evaluated
            s_0 (float): $s$ point where the phase shift is extracted
            Smatrix_abs_tol : absolute tolerance for deviation of real part of S-matrix amplitudes
                from 1, used as criteria to stop calculation ig higher partial waves are negligble

        Returns:
            sae (ScatteringAmplitudeEmulator): scattering amplitude emulator

        """
        if base_solver is None:
            base_solver = SchroedingerEquation.make_base_solver()

        if l_max is None:
            l_max = interaction_space.l_max

        bases = []
        for interaction_list in tqdm(interaction_space.interactions):
            basis_list = []
            for interaction in interaction_list:
                solver = base_solver.clone_for_new_interaction(interaction)
                if s_mesh is None:
                    s_mesh = np.linspace(
                        solver.domain[0],
                        solver.domain[1],
                        SchroedingerEquation.DEFAULT_NUM_PTS,
                    )
                basis_list.append(
                    RelativeBasis(
                        solver,
                        alpha_train,
                        s_mesh,
                        n_basis,
                        use_svd=use_svd,
                        **basis_kwargs,
                    )
                )
            bases.append(basis_list)

        return cls(
            interaction_space,
            bases,
            l_max,
            angles=angles,
            s_0=base_solver.s_0,
            Smatrix_abs_tol=Smatrix_abs_tol,
        )

    def __init__(
        self,
        interaction_space: InteractionSpace,
        bases: list,
        l_max: int = None,
        angles: np.array = DEFAULT_ANGLE_MESH,
        s_0: float = 6 * np.pi,
        verbose: bool = True,
        Smatrix_abs_tol=1.0e-6,
        initialize_emulator=True,
    ):
        r"""Trains a reduced-basis emulator that computes differential and total cross sections
            (from emulated phase shifts).

        Parameters:
            interaction_space (InteractionSpace): local interaction up to (and including $\ell_\max$)
            bases (list[Basis]): list of `Basis` objects
            l_max (int): maximum angular momentum to include in the sum approximating the cross section
            angles (ndarray): Differential cross sections are functions of the
                angles. These are the specific values at which the user wants to
                emulate the cross section.
            s_0 (float): $s$ point where the phase shift is extracted
            verbose (bool): Do you want the class to print out warnings?
            Smatrix_abs_tol : absolute tolerance for deviation of real part of S-matrix amplitudes
                from 1, used as criteria to stop calculation ig higher partial waves are negligble
            initialize_emulator : build the low-order emulator (True required for emulation)

        Attributes:
            l_max (int): maximum angular momentum
            angles (ndarray): angle values at which the differential cross section is desired
            rbes (list): list of `ReducedBasisEmulators`; one for each partial wave
                (and total $j$ with spin-orbit)
            ls (ndarray): angular momenta; shape = (`l_max`+1, 1)
            P_l_costheta (ndarray): Legendre polynomials evaluated at `angles`
            P_1_l_costheta (ndarray): **associated** Legendre polynomials evalated at `angles`
            k_c (float): Coulomb momentum, $k\eta$
            eta (float): Sommerfeld parameter
            sigma_l (float): Coulomb phase shift
            f_c (ndarray): scattering amplitude
            rutherford (ndarray): Rutherford scattering

        """
        # partial waves
        if l_max is None:
            l_max = interaction_space.l_max
        self.l_max = l_max
        self.Smatrix_abs_tol = Smatrix_abs_tol

        # construct bases
        self.rbes = []
        for interaction_list, basis_list in zip(interaction_space.interactions, bases):
            self.rbes.append(
                [
                    ReducedBasisEmulator(
                        interaction,
                        basis,
                        s_0=s_0,
                        initialize_emulator=initialize_emulator,
                    )
                    for (interaction, basis) in zip(interaction_list, basis_list)
                ]
            )

        # Let's precompute the things we can.
        self.angles = angles.copy()
        self.ls = np.arange(self.l_max + 1)[:, np.newaxis]
        self.P_l_costheta = eval_legendre(self.ls, np.cos(self.angles))
        self.P_1_l_costheta = np.array(
            [eval_assoc_legendre(l, np.cos(self.angles)) for l in self.ls]
        )

        # Coulomb scattering amplitude
        if (
            self.rbes[0][0].interaction.k is not None
            and self.rbes[0][0].interaction.k_c > 0
        ):
            k = self.rbes[0][0].interaction.k
            self.k_c = self.rbes[0][0].interaction.k_c
            self.eta = self.k_c / k
            self.sigma_l = np.angle(gamma(1 + self.ls + 1j * self.eta))
            sin2 = np.sin(self.angles / 2) ** 2
            self.f_c = (
                -self.eta
                / (2 * k * sin2)
                * np.exp(
                    2j * self.sigma_l[0]
                    - 2j * self.eta * np.log(np.sin(self.angles / 2))
                )
            )
            self.rutherford = (
                10 * self.eta**2 / (4 * k**2 * np.sin(self.angles / 2) ** 4)
            )
        else:
            self.sigma_l = np.angle(gamma(1 + self.ls + 1j * 0))
            self.k_c = 0
            self.eta = 0
            self.f_c = 0.0 * np.exp(2j * self.sigma_l[0])
            self.rutherford = 0.0 / (np.sin(self.angles / 2) ** 4)

    def emulate_phase_shifts(self, alpha: np.array):
        r"""Gives the phase shifts for each partial wave.  Order is [l=0, l=1,
            ..., l=l_max-1].

        Parameters:
            alpha (ndarray): parameter-space vector

        Returns:
            phase_shift (list): emulated phase shifts

        """
        return [
            [rbe.emulate_phase_shift(alpha) for rbe in rbe_list]
            for rbe_list in self.rbes
        ]

    def exact_phase_shifts(self, alpha: np.array):
        r"""Gives the phase shifts for each partial wave. Order is [l=0, l=1,
            ..., l=l_max-1].

        Parameters:
            alpha (ndarray): parameter-space vector

        Returns:
            phase_shift (list): high-fidelity phase shifts

        """
        return [
            [rbe.basis.solver.delta(alpha) for rbe in rbe_list]
            for rbe_list in self.rbes
        ]

    def emulate_dsdo(self, alpha: np.array):
        r"""Emulates the differential cross section (dsigma/dOmega = dsdo) in mb/Sr.

        Parameters:
            alpha (ndarray): parameter-space vector

        Returns:
            dsdo (ndarray): emulated differential cross section

        """
        Splus, Sminus = self.emulate_smatrix_elements(alpha)
        return self.dsdo(alpha, Splus, Sminus)

    def exact_dsdo(self, alpha: np.array):
        r"""Calculates the high-fidelity differential cross section (dsigma/dOmega = dsdo) in mb/Sr.

        Parameters:
            alpha (ndarray): parameter-space vector

        Returns:
            dsdo (ndarray): high-fidelity differential cross section

        """
        Splus, Sminus = self.exact_smatrix_elements(alpha)
        return self.dsdo(alpha, Splus, Sminus)

    def emulate_xs(self, alpha: np.array, angles: np.array = None):
        r"""Emulates the:
            - differential cross section in mb/Sr (as a ratio to a Rutherford xs if provided)
            - analyzing power
            - total and reacion cross sections in mb

            Paramaters:
                alpha (ndarray) : interaction parameters
                angles (ndarray) : (optional), angular grid on which to evaluate analyzing \
                powers and differential cross section

            Returns :
                cross sections (NucleonNucleusXS) :
        """
        # get phase shifts and wavenumber
        Splus, Sminus = self.emulate_smatrix_elements(alpha)
        return self.calculate_xs(Splus, Sminus, alpha, angles)

    def exact_xs(self, alpha: np.array, angles: np.array = None):
        r"""Calculates the exact:
            - differential cross section in mb/Sr (as a ratio to a Rutherford xs if provided)
            - analyzing power
            - total and reacion cross sections in mb

            Paramaters:
                alpha (ndarray) : interaction parameters
                angles (ndarray) : (optional), angular grid on which to evaluate analyzing \
                powers and differential cross section

            Returns :
                cross sections (NucleonNucleusXS) :
        """
        # get phase shifts and wavenumber
        Splus, Sminus = self.exact_smatrix_elements(alpha)
        return self.calculate_xs(Splus, Sminus, alpha, angles)

    def emulate_wave_functions(self, alpha: np.array):
        r"""Gives the wave functions for each partial wave.  Returns a list of
            arrays.  Order is [l=0, l=1, ..., l=l_max-1].

        Parameters:
            alpha (ndarray): parameter-space vector

        Returns:
            wave_functions (list): emulated wave functions


        """
        return [[x.emulate_wave_function(alpha) for x in rbe] for rbe in self.rbes]

    def exact_wave_functions(
        self, alpha: np.array, s_mesh: np.array = None, **solver_kwargs
    ):
        r"""Gives the wave functions for each partial wave.  Returns a list of
            arrays.  Order is [l=0, l=1, ..., l=l_max-1].

        Parameters:
            alpha (ndarray): parameter-space vector
            s_mesh (ndarray): s_mesh on which to evaluate phi, if different from the one used
                for emulation
            solver_kwargs (ndarray): passed to SchroedingerEquation.phi

        Returns:
            wave_functions (list): emulated wave functions


        """
        if s_mesh is None:
            return [
                [x.basis.solver.phi(alpha, x.s_mesh, **solver_kwargs) for x in rbe]
                for rbe in self.rbes
            ]
        else:
            return [
                [x.basis.solver.phi(alpha, s_mesh, **solver_kwargs) for x in rbe]
                for rbe in self.rbes
            ]

    def dsdo(self, alpha: np.array, Splus: np.array, Sminus: np.array):
        r"""Gives the differential cross section (dsigma/dOmega = dsdo) in mb/Sr.

        Parameters:
            alpha (ndarray): parameter-space vector
            Splus (ndarray): spin-up smatrix elements
            Sminus (ndarray): spin-down smatrix elements

        Returns:
            dsdo (ndarray): differential cross section (fm^2)

        """
        k = self.rbes[0][0].interaction.momentum(alpha)

        Splus = Splus[:, np.newaxis]
        Sminus = Sminus[:, np.newaxis]
        lmax = Splus.shape[0]
        l = self.ls[:lmax]

        A = self.f_c + (1 / (2j * k)) * np.sum(
            np.exp(2j * self.sigma_l[:lmax])
            * ((l + 1) * (Splus - 1) + l * (Sminus - 1))
            * self.P_l_costheta[:lmax, ...],
            axis=0,
        )
        B = (1 / (2j * k)) * np.sum(
            np.exp(2j * self.sigma_l[:lmax])
            * (Splus - Sminus)
            * self.P_1_l_costheta[:lmax, ...],
            axis=0,
        )

        dsdo = 10 * (np.conj(A) * A + np.conj(B) * B).real
        if self.k_c > 0:
            return dsdo / self.rutherford
        else:
            return dsdo

    def exact_smatrix_elements(self, alpha):
        r"""Returns:
        Sl_plus (ndarray) : l-s aligned S-matrix elements for partial waves up to where
            Splus.real - 1 < Smatrix_abs_tol
        Sl_minus (ndarray) : same as Splus, but l-s anti-aligned
        """

        Splus = np.zeros(self.l_max, dtype=np.complex128)
        Sminus = np.zeros(self.l_max, dtype=np.complex128)
        Splus[0] = self.rbes[0][0].basis.solver.smatrix(alpha)
        Sminus[0] = Splus[0]
        for l in range(1, self.l_max):
            Splus[l] = self.rbes[l][0].basis.solver.smatrix(alpha)
            Sminus[l] = self.rbes[l][1].basis.solver.smatrix(alpha)
            if (
                np.absolute(Splus[l] - 1) < self.Smatrix_abs_tol
                and np.absolute(Sminus[l] - 1) < self.Smatrix_abs_tol
            ):
                break

        return Splus[:l], Sminus[:l]

    def exact_rmatrix_elements(self, alpha):
        r"""Returns:
        Rl_plus (ndarray) : l-s aligned R-matrix elements for partial waves up to where
        Rl_minus (ndarray) : same as Splus, but l-s anti-aligned
        """
        Rplus = np.array(
            [
                rbe_list[0].basis.solver.rmatrix(alpha)
                for rbe_list in self.rbes[0 : self.l_max]
            ]
        )
        Rminus = np.array(
            [Rplus[0]]
            + [
                rbe_list[1].basis.solver.rmatrix(alpha)
                for rbe_list in self.rbes[1 : self.l_max]
            ]
        )
        return Rplus, Rminus

    def emulate_rmatrix_elements(self, alpha):
        r"""Returns:
        Rl_plus (ndarray) : l-s aligned R-matrix elements for partial waves
        Rl_minus (ndarray) : same as Splus, but l-s anti-aligned
        """
        Rplus = np.array(
            [
                rbe_list[0].R_matrix_element(alpha)
                for rbe_list in self.rbes[0 : self.l_max]
            ]
        )
        Rminus = np.array(
            [Rplus[0]]
            + [
                rbe_list[1].R_matrix_element(alpha)
                for rbe_list in self.rbes[1 : self.l_max]
            ]
        )
        return Rplus, Rminus

    def emulate_smatrix_elements(self, alpha):
        r"""Returns:
        Sl_plus (ndarray) : l-s aligned S-matrix elements for partial waves up to where
            Splus < Smatrix_abs_tol
        Sl_minus (ndarray) : same as Splus, but l-s anti-aligned
        """
        Splus = np.zeros(self.l_max, dtype=np.complex128)
        Sminus = np.zeros(self.l_max, dtype=np.complex128)
        Splus[0] = self.rbes[0][0].S_matrix_element(alpha)
        Sminus[0] = Splus[0]
        for l in range(1, self.l_max):
            Splus[l] = self.rbes[l][0].S_matrix_element(alpha)
            Sminus[l] = self.rbes[l][1].S_matrix_element(alpha)
            if (
                np.absolute(Splus[l]) < self.Smatrix_abs_tol
                and np.absolute(Sminus[l]) < self.Smatrix_abs_tol
            ):
                break

        return Splus[:l], Sminus[:l]

    def calculate_xs(
        self,
        Splus: np.array,
        Sminus: np.array,
        alpha: np.array,
        angles: np.array = None,
    ):
        r"""Calculates the:
            - differential cross section in mb/Sr (as a ratio to a Rutherford xs if provided)
            - analyzing power
            - total and reacion cross sections in mb

            Paramaters:
                Splus (ndarray) : spin-up phase shifs
                Sminus (ndarray) : spin-down phase shifts
                alpha (ndarray) : interaction parameters
                angles (ndarray) : (optional), angular grid on which to evaluate analyzing \
                powers and differential cross section

            Returns :
                cross sections (NucleonNucleusXS) :
        """
        k = self.rbes[0][0].interaction.momentum(alpha)

        # determine desired angle grid and precompute
        # Legendre functions if necessary
        if angles is None:
            angles = self.angles
            P_l_costheta = self.P_l_costheta
            P_1_l_costheta = self.P_1_l_costheta
            rutherford = self.rutherford
            f_c = self.f_c
        else:
            assert np.max(angles) <= np.pi and np.min(angles) >= 0
            P_l_costheta = np.array(
                [eval_legendre(l, np.cos(angles)) for l in range(self.l_max)]
            )
            P_1_l_costheta = np.array(
                [eval_assoc_legendre(l, np.cos(angles)) for l in range(self.l_max)]
            )
            sin2 = np.sin(angles / 2) ** 2
            rutherford = 10 * self.eta**2 / (4 * k**2 * sin2**2)
            f_c = (
                -self.eta
                / (2 * k * sin2)
                * np.exp(-1j * self.eta * np.log(sin2) + 2j * self.sigma_l[0])
            )

        if self.rbes[0][0].interaction.eta(alpha) > 0:
            return NucleonNucleusXS(
                *xs_calc_coulomb(
                    k,
                    angles,
                    Splus,
                    Sminus,
                    P_l_costheta,
                    P_1_l_costheta,
                    f_c,
                    self.sigma_l,
                    rutherford,
                )
            )
        else:
            return NucleonNucleusXS(
                *xs_calc_neutral(
                    k,
                    angles,
                    Splus,
                    Sminus,
                    P_l_costheta,
                    P_1_l_costheta,
                )
            )

    def percent_explained_variance(self):
        r"""
        Returns:
            (float) : percent of variance explained in the training set by the first n_basis principal
            components
        """
        return [
            [rbe.basis.percent_explained_variance() for rbe in rbe_list]
            for rbe_list in self.rbes
        ]

    def save(self, filename):
        r"""Saves the emulator to the desired file.

        Parameters:
            filename (string): name of file

        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
