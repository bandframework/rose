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
        solver: jitr.rmatrix.Solver,
    ):
        self.s_0 = s_0
        self.domain = [0, s_0]
        self.interaction = interaction
        self.solver = solver

        if self.interaction is not None:
            if self.interaction.k_c == 0:
                eta = 0
            else:
                # There is Coulomb, but we haven't (yet) worked out how to emulate
                # across energies, so we can precompute H+ and H- stuff.
                eta = self.interaction.eta(None)

        self.Hm = H_minus(self.s_0, self.interaction.ell, eta)
        self.Hp = H_plus(self.s_0, self.interaction.ell, eta)
        self.Hmp = H_minus_prime(self.s_0, self.interaction.ell, eta)
        self.Hpp = H_plus_prime(self.s_0, self.interaction.ell, eta)

        self.weight = np.ones(1)
        self.l = np.array([interaction.ell])
        self.a = s_0
        self.eta = np.array([eta])
        self.couplings = np.array([[0.0]])

        self.asym = jitr.reactions.Asymptotics(
            np.array([self.Hp]),
            np.array([self.Hm]),
            np.array([self.Hpp]),
            np.array([self.Hmp]),
        )

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

        # these are always parameter independent - we can precompute them
        self.basis_boundary = self.solver.precompute_boundaries(self.a)
        self.free_matrix = self.solver.free_matrix(self.a, self.l)

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
        ch = jitr.reactions.Channels(
            np.array([self.interaction.E(alpha)]),
            np.array([self.interaction.momentum(alpha)]),
            np.array([self.interaction.reduced_mass(alpha)]),
            self.eta,
            self.a,
            self.l,
            self.couplings,
        )

        return ch

    def phi(
        self,
        alpha: np.array,
        s_mesh: np.array,
        **kwargs,
    ):
        assert s_mesh[-1] <= self.s_0

        ch = self.get_channel_info(alpha)
        R, S, x, uext_prime_boundary = self.solver.solve(
            ch,
            self.asym,
            local_interaction=self.potential,
            local_args=self.get_args(alpha),
            free_matrix=self.free_matrix,
            basis_boundary=self.basis_boundary,
            wavefunction=True,
        )
        return jitr.reactions.wavefunction.Wavefunctions(
            self.solver,
            x,
            S,
            uext_prime_boundary,
            ch,
            incoming_weights=self.weight,
        ).uint()[0](s_mesh)

    def smatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        ch = self.get_channel_info(alpha)
        R, S, x, uext_prime_boundary = self.solver.solve(
            ch,
            self.asym,
            local_interaction=self.potential,
            local_args=self.get_args(alpha),
            free_matrix=self.free_matrix,
            basis_boundary=self.basis_boundary,
            wavefunction=False,
        )
        return S[0, 0]

    def rmatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        ch = self.get_channel_info(alpha)
        ch = self.get_channel_info(alpha)
        R, S, x, uext_prime_boundary = self.solver.solve(
            ch,
            self.asym,
            local_interaction=self.potential,
            local_args=self.get_args(alpha),
            free_matrix=self.free_matrix,
            basis_boundary=self.basis_boundary,
            wavefunction=False,
        )
        return R[0, 0]
