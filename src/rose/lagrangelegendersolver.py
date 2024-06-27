from .schroedinger import SchroedingerEquation
from .interaction import Interaction
from .constants import HBARC
from .njit_solver_utils import potential

import numpy as np
import jitr


class LagrangeRmatrix(SchroedingerEquation):
    r"""Implements a ROSE HF solver for an interaction with defined energy and l using jitr."""

    def __init__(
        self,
        interaction: Interaction,
        s_0,
        Nbasis: int,
    ):
        self.s_0 = s_0
        self.domain = [0, s_0]
        self.interaction = interaction

        mu = self.interaction.mu
        if mu is None:
            mu = 0.0

        self.sys = jitr.ProjectileTargetSystem(
            np.array([mu]),
            np.array([s_0]),
            np.array([interaction.ell]),
            interaction.Z_1,
            interaction.Z_2,
            1,
        )

        self.solver = jitr.LagrangeRMatrixSolver(
            Nbasis,
            1,
            self.sys,
            ecom=interaction.energy,
            asym=jitr.CoulombAsymptotics,
        )

    def clone_for_new_interaction(self, interaction: Interaction):
        return LagrangeRmatrix(interaction, self.s_0, self.solver.kernel.nbasis)

    def get_channel_info(self, alpha):
        ch = jitr.ChannelData(
            self.interaction.ell,
            self.interaction.reduced_mass(alpha),
            self.s_0,
            self.interaction.E(alpha),
            self.interaction.momentum(alpha),
            self.interaction.eta(alpha),
        )

        im = jitr.InteractionMatrix(1)
        im.set_local_interaction(potential)
        im.local_args[0, 0] = (
            alpha,
            self.interaction.Z_1 * self.interaction.Z_2,
            self.interaction.coulomb_cutoff(alpha),
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
            self.interaction.spin_orbit_term.l_dot_s,
        )

        return [ch], im

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
            wavefunction=True,
        )
        return jitr.Wavefunctions(
            self.solver,
            x,
            S,
            uext_prime_boundary,
            self.sys.incoming_weights,
            [ch],
            self.solver.asym,
        ).uint()[0](s_mesh)

    def smatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        ch, im = self.get_channel_info(alpha)
        R, S, uext_prime_boundary = self.solver.solve(im, ch)
        return S[0, 0]

    def rmatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        ch, im = self.get_channel_info(alpha)
        R, S, uext_prime_boundary = self.solver.solve(im, ch)
        return R[0, 0]
