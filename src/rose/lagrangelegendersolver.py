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
        Nbasis: int,
    ):
        assert sys.nchannels == 1
        self.interaction = interaction
        self.sys = jitr.ProjectileTargetSystem(
            np.array([interaction.mu]),
            np.array([interaction.s_0]),
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

        self.domain = [0, self.sys.channel_radii[0]]
        self.s_0 = self.domain[1]
        self.param_mask = np.ones(self.interaction.n_theta, dtype=bool)
        if self.interaction.energy is None:
            self.param_mask[0] = False
        if self.interaction.mu is None:
            self.param_mask[1] = False


    def clone_for_new_interaction(self, interaction: Interaction):
        return LagrangeRmatrix(interaction, self.sys, self.solver.kernel.nbasis)

    def get_channel_info(self, alpha):
        ch = jitr.ChannelData(
            self.interaction.ell,
            self.s_0,
            self.interaction.E(alpha),
            self.interaction.momentum(alpha),
            self.interaction.eta(alpha),
        )

        im = jitr.InteractionMatrix(1)
        im.set_local_interaction(potential)
        im.local_args[0, 0] = self.interaction.bundle_gcoeff_args(alpha)

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
        R, S, uext_prime_boundary = self.solver.solve(im, [ch])
        return S[0, 0]

    def rmatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        ch, im = self.get_channel_info(alpha)
        R, S, uext_prime_boundary = self.solver.solve(im, [ch])
        return R[0, 0]
