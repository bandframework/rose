from .schroedinger import SchroedingerEquation
from .interaction import Interaction
from .constants import HBARC

import numpy as np
import jitr


class LagrangeRmatrix(SchroedingerEquation):
    r"""Implements a ROSE HF solver for an interaction with defined energy and l using jitr."""

    def __init__(
        self,
        interaction: Interaction,
        sys: jitr.ProjectileTargetSystem,
        Nbasis: int,
    ):
        assert sys.nchannels == 1
        self.interaction = interaction
        self.sys = jitr.ProjectileTargetSystem(
            sys.reduced_mass,
            sys.channel_radii,
            np.array([interaction.ell]),
            sys.Ztarget,
            sys.Zproj,
            1,
        )

        self.ch = self.sys.build_channels(interaction.energy)
        self.solver = jitr.LagrangeRMatrixSolver(
            Nbasis,
            1,
            self.sys,
            ecom=interaction.energy,
            channel_matrix=self.ch,
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

    def phi(
        self,
        alpha: np.array,
        s_mesh: np.array,
        **kwargs,
    ):
        if not self.param_mask[1]:
            mu = alpha[1]
        else:
            mu = sys.reduced_mass[0]
        if not self.param_mask[0]:
            energy = alpha[0]
        else:
            energy = self.ch.E

        ch = jitr.ChannelData(
            self.interaction.ell,
            mu,
            self.s_0,
            energy,
            np.sqrt(2 * energy * mu) / HBARC,
            0.0,
            self.domain,
        )

        im = jitr.InteractionMatrix(1)
        im.set_local_interaction(potential)
        im.local_args[0, 0] = (
            self.interaction.ell,
            self.interaction.spin_orbit_term.l_dot_s,
            alpha,
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
        )
        R, S, x, uext_prime_boundary = self.solver.solve(
            im,
            [ch],
            wavefunction=True,
        )
        return jitr.Wavefunctions(
            self.solver,
            x,
            S,
            uext_prime_boundary,
            self.sys.incoming_weights,
            self.ch,
            self.solver.asym,
        ).uint()[0](s_mesh)

    def smatrix(
        self,
        alpha: np.array,
        s_0: float = None,
        **kwargs,
    ):
        r"""
        Ignores s_0
        """
        if not self.param_mask[1]:
            mu = alpha[1]
        else:
            mu = sys.reduced_mass[0]
        if not self.param_mask[0]:
            energy = alpha[0]
        else:
            energy = self.ch.E

        ch = jitr.ChannelData(
            self.interaction.ell,
            mu,
            self.s_0,
            energy,
            np.sqrt(2 * energy * mu) / HBARC,
            0.0,
            self.domain,
        )
        im = jitr.InteractionMatrix(1)
        im.set_local_interaction(potential)
        im.local_args[0, 0] = (
            self.interaction.ell,
            self.interaction.spin_orbit_term.l_dot_s,
            alpha,
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
        )
        R, S, uext_prime_boundary = self.solver.solve(im, [ch])
        return S[0, 0]

    def rmatrix(
        self,
        alpha: np.array,
        s_0: float = None,
        **kwargs,
    ):
        r"""
        Ignores s_0
        """
        if not self.param_mask[1]:
            mu = alpha[1]
        else:
            mu = sys.reduced_mass[0]
        if not self.param_mask[0]:
            energy = alpha[0]
        else:
            energy = self.ch.E

        ch = jitr.ChannelData(
            self.interaction.ell,
            mu,
            self.s_0,
            energy,
            np.sqrt(2 * energy * mu) / HBARC,
            0.0,
            self.domain,
        )
        im = jitr.InteractionMatrix(1)
        im.set_local_interaction(potential)
        im.local_args[0, 0] = (
            self.interaction.ell,
            self.interaction.spin_orbit_term.l_dot_s,
            alpha,
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
        )
        R, S, uext_prime_boundary = self.solver.solve(im, [ch])
        return R[0, 0]
