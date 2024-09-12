from .schroedinger import SchroedingerEquation
from .interaction import Interaction
from .energized_interaction_eim import EnergizedInteractionEIM
from .utility import potential, potential_plus_coulomb
from .free_solutions import H_minus, H_plus, H_minus_prime, H_plus_prime

import numpy as np
from jitr import rmatrix, reactions


class LagrangeRmatrix(SchroedingerEquation):
    r"""Implements a ROSE HF solver for an interaction with defined energy and l using jitr."""

    def __init__(
        self,
        interaction: Interaction,
        s_0,
        solver: rmatrix.Solver
    ):
        self.l = np.array([interaction.ell])
        self.a = s_0
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
        self.asymptotics = reactions.system.Asymptotics(
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
            alpha[self.param_offset:],
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
            self.interaction.spin_orbit_term.l_dot_s,
        )

    def get_args_coulomb(self, alpha):
        return (
            alpha[self.param_offset:],
            self.interaction.Z_1 * self.interaction.Z_2,
            self.interaction.coulomb_cutoff(alpha),
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
            self.interaction.spin_orbit_term.l_dot_s,
        )

    def clone_for_new_interaction(self, interaction: Interaction):
        return LagrangeRmatrix(interaction, self.s_0, self.solver)

    def get_channel_info(self, alpha):
        E = self.interaction.E(alpha)
        mu = self.interaction.reduced_mass(alpha)
        k = self.interaction.momentum(alpha)
        eta = self.interaction.eta(alpha)
        ch = reactions.system.Channels(
            np.array([E]),
            np.array([k]),
            np.array([mu]),
            np.array([eta]),
            self.a,
            self.l,
            np.array([[1]]),
        )

        return ch, self.get_args(alpha)

    def phi(
        self,
        alpha: np.array,
        s_mesh: np.array,
        **kwargs,
    ):
        assert s_mesh[-1] <= self.s_0

        ch, args = self.get_channel_info(alpha)
        R, S, x, uext_prime_boundary = self.solver.solve(
            ch,
            self.asymptotics,
            local_interaction=self.potential,
            local_args=args,
            free_matrix=self.free_matrix,
            basis_boundary=self.basis_boundary,
            wavefunction=True,
        )
        return reactions.wavefunction.Wavefunctions(
            self.solver,
            x,
            S,
            uext_prime_boundary,
            ch,
        ).uint()[0](s_mesh)

    def smatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        ch, args = self.get_channel_info(alpha)
        R, S, uext_prime_boundary = self.solver.solve(
            ch,
            self.asymptotics,
            local_interaction=self.potential,
            local_args=args,
            free_matrix=self.free_matrix,
            basis_boundary=self.basis_boundary,
        )
        return S[0, 0]

    def rmatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        ch, args = self.get_channel_info(alpha)
        R, S, uext_prime_boundary = self.solver.solve(
            ch,
            self.asymptotics,
            local_interaction=self.potential,
            local_args=args,
            free_matrix=self.free_matrix,
            basis_boundary=self.basis_boundary,
        )
        return R[0, 0]
