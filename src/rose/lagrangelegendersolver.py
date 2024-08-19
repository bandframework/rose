from .schroedinger import SchroedingerEquation
from .interaction import Interaction
from .energized_interaction_eim import EnergizedInteractionEIM
from .utility import potential, potential_plus_coulomb
from .free_solutions import H_minus, H_plus, H_minus_prime, H_plus_prime

import numpy as np
import jitr


class LagrangeRmatrix(SchroedingerEquation):
    r"""Implements a ROSE HF solver for an interaction with defined energy and l using jitr."""

    def __init__(
        self,
        interaction: Interaction,
        s_0,
        solver: jitr.RMatrixSolver,
    ):
        l = np.array([interaction.ell])
        a = np.array([s_0])
        self.s_0 = s_0
        self.domain = [0, s_0]
        self.interaction = interaction
        self.solver = solver

        if self.interaction is not None:
            if self.interaction.k_c == 0:
                self.eta = 0
            else:
                # There is Coulomb, but we haven't (yet) worked out how to emulate
                # across energies, so we can precompute H+ and H- stuff.
                self.eta = self.interaction.eta(None)

        self.Hm = H_minus(self.s_0, self.interaction.ell, self.eta)
        self.Hp = H_plus(self.s_0, self.interaction.ell, self.eta)
        self.Hmp = H_minus_prime(self.s_0, self.interaction.ell, self.eta)
        self.Hpp = H_plus_prime(self.s_0, self.interaction.ell, self.eta)
        self.channels = np.zeros(1, dtype=jitr.channel_dtype)
        self.channels["weight"] = np.ones(1)
        self.channels["l"] = l
        self.channels["a"] = a
        self.channels["eta"] = self.eta
        self.channels["Hp"] = np.array([self.Hp])
        self.channels["Hm"] = np.array([self.Hm])
        self.channels["Hpp"] = np.array([self.Hpp])
        self.channels["Hmp"] = np.array([self.Hmp])

        # for Energized EIM interactions, E, mu k take up the first 3 spots
        # in the parameter vector, so we offset to account for that
        self.param_offset = 0
        if isinstance(self.interaction, EnergizedInteractionEIM):
            self.param_offset = 3

        if self.interaction.Z_1 * self.interaction.Z_2 > 0:
            self.potential = potential_plus_coulomb
            self.get_args = self.get_args_coulomb
        else:
            self.potential = potential
            self.get_args = self.get_args_neutral

        self.im = jitr.InteractionMatrix(1)
        self.im.set_local_interaction(self.potential)

        # these are always parameter independent - we can precompute them
        self.basis_boundary = self.solver.precompute_boundaries(a)
        self.free_matrix = self.solver.free_matrix(a, l)

    def get_args_neutral(self, alpha):
        return (
            alpha[self.param_offset :],
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
            self.interaction.spin_orbit_term.l_dot_s,
        )

    def get_args_coulomb(self, alpha):
        return (
            alpha[self.param_offset :],
            self.interaction.Z_1 * self.interaction.Z_2,
            self.interaction.coulomb_cutoff(alpha),
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
            self.interaction.spin_orbit_term.l_dot_s,
        )

    def clone_for_new_interaction(self, interaction: Interaction):
        return LagrangeRmatrix(interaction, self.s_0, self.solver)

    def get_channel_info(self, alpha):
        self.im.local_args[0, 0] = self.get_args(alpha)
        self.channels["E"] = self.interaction.E(alpha)
        self.channels["mu"] = self.interaction.reduced_mass(alpha)
        self.channels["k"] = self.interaction.momentum(alpha)

        return self.channels, self.im

    def phi(
        self,
        alpha: np.array,
        s_mesh: np.array,
        **kwargs,
    ):
        assert s_mesh[-1] <= self.s_0

        ch, im = self.get_channel_info(alpha)
        R, S, x, uext_prime_boundary = self.solver.solve(
            im,
            ch,
            free_matrix=self.free_matrix,
            basis_boundary=self.basis_boundary,
            wavefunction=True,
        )
        return jitr.Wavefunctions(
            self.solver,
            x,
            S,
            uext_prime_boundary,
            self.channels["weight"],
            jitr.make_channel_data(ch),
        ).uint()[0](s_mesh)

    def smatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        ch, im = self.get_channel_info(alpha)
        R, S, uext_prime_boundary = self.solver.solve(
            im,
            ch,
            free_matrix=self.free_matrix,
            basis_boundary=self.basis_boundary,
        )
        return S[0, 0]

    def rmatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        ch, im = self.get_channel_info(alpha)
        R, S, uext_prime_boundary = self.solver.solve(
            im,
            ch,
            free_matrix=self.free_matrix,
            basis_boundary=self.basis_boundary,
        )
        return R[0, 0]
