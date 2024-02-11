from .schroedinger import SchroedingerEquation
from .interaction import Interaction

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

    def clone_for_new_interaction(self, interaction: Interaction):
        return LagrangeRmatrix1ch(interaction, self.sys, self.solver.kernel.nbasis)

    def phi(
        self,
        alpha: np.array,
        s_mesh: np.array,
        **kwargs,
    ):
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
            self.ch,
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
        im = jitr.InteractionMatrix(1)
        im.set_local_interaction(potential)
        im.local_args[0, 0] = (
            self.interaction.ell,
            self.interaction.spin_orbit_term.l_dot_s,
            alpha,
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
        )
        R, S, uext_prime_boundary = self.solver.solve(im, self.ch)
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
        im = jitr.InteractionMatrix(1)
        im.set_local_interaction(potential)
        im.local_args[0, 0] = (
            self.interaction.ell,
            self.interaction.spin_orbit_term.l_dot_s,
            alpha,
            self.interaction.v_r,
            self.interaction.spin_orbit_term.v_so,
        )
        R, S, uext_prime_boundary = self.solver.solve(im, self.ch)
        return R[0, 0]
