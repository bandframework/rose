r"""
Some helpful utilities for training an emulator and visualizing performance
"""
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
import seaborn as sns
from tqdm import tqdm

from .schroedinger import SchroedingerEquation
from .basis import Basis, CustomBasis
from .interaction import Interaction, InteractionSpace
from .interaction_eim import InteractionEIM, InteractionEIMSpace
from .energized_interaction_eim import EnergizedInteractionEIMSpace
from .scattering_amplitude_emulator import ScatteringAmplitudeEmulator
from .utility import latin_hypercube_sample

# some colors Pablo likes for plotting
colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def sample_params_LHC(
    N: int,
    central_vals: np.array = None,
    scale: float = None,
    seed: int = None,
    bounds: np.array = None,
):
    r"""
    Sampling parameters from a finite box in parameter space around some central values using the Latin hypercube method
    Parameters:
        N : number of samples
        central_vals : central values of each parameter
        scale : fraction of central vals, such that (1 +/- scale)*abs(central_vals) defines the bounds
                of the box
        seed : RNG seed. If None, uses entropy from the system
    Returns:
        (ndarray) : N samples
    """
    if bounds is None:
        assert scale is not None
        assert central_vals is not None
        bounds = np.array(
            [
                central_vals - np.fabs(central_vals * scale),
                central_vals + np.fabs(central_vals * scale),
            ]
        ).T

    return latin_hypercube_sample(N, bounds, seed)


def build_sae_config_set(
    sae_configs: list,
    base_interaction: InteractionSpace,
    theta_train: np.array,
    bounds=None,
    **SAE_kwargs,
):
    r"""
    build a list of `ScatteringAmplitudeEmulator`s to the specification of `sae_configs`
    Parameters:
        base_interaction (InteractionSpace): the space of Interactions to replicate for each config
    """

    # first build the largest one, which has all the bases we need
    imax = np.argmax([conf[0] for conf in sae_configs])
    saes = []
    inters = []
    corresponding_inter, biggest_sae = build_sae(
        sae_configs[imax],
        base_interaction,
        theta_train,
        bounds,
        **SAE_kwargs,
    )

    if "base_solver" in SAE_kwargs:
        SAE_kwargs.pop("base_solver")

    # get the largest set of bases (for each partial wave)
    bases = [[rbe.basis for rbe in rbe_list] for rbe_list in biggest_sae.rbes]

    # now do the rest using the bases we've built
    for i in tqdm(range(len(sae_configs))):
        if i == imax:
            saes.append(biggest_sae)
            inters.append(corresponding_inter)
        else:
            inter, sae = build_sae(
                sae_configs[i],
                base_interaction,
                theta_train,
                bounds,
                bases,
                **SAE_kwargs,
            )
            saes.append(sae)
            inters.append(inter)

    return inters, saes


def build_sae(
    sae_config: tuple,
    base_interaction,
    theta_train: np.array = None,
    bounds=None,
    bases: list = None,
    **SAE_kwargs,
):
    r"""
    build an EIM emulator to specification of sae_config, using base_interaction
    Parameters:
        sae_config :  (size of reduced basis, number of EIM terms)
        base_interaction :  interaction on which to train
        bases (optional) : if a full set of bases for each interaction has been solved
            for already, re-use basis.vectors rather than re-calculating them
        theta_train (optional) : if bases is not provided, simply pass in the
            training samples and re-train the emulator
        SAE_kwargs : passed to ScatteringAmplitudeEmulator. If bases is None, passed to
            ScatteringAmplitudeEmulator.from_train
    """

    (n_basis, n_EIM) = sae_config

    base = base_interaction.interactions[0][0]

    if bounds is not None:
        eim_explicit_training = False
        eim_training_info = bounds
    else:
        eim_explicit_training = True
        eim_training_info = theta_train

    interactions = InteractionSpace(
        coordinate_space_potential=base.v_r,
        n_theta=base.n_theta,
        mu=base.mu,
        energy=base.energy,
        training_info=eim_training_info,
        Z_1=base.Z_1,
        Z_2=base.Z_2,
        R_C=base.R_C,
        is_complex=base.is_complex,
        spin_orbit_term=base.spin_orbit_term.v_so,
        explicit_training=eim_explicit_training,
        rho_mesh=base.s_mesh,
        n_basis=n_EIM,
        n_train=base.n_train,
        method=base.method,
        l_max=base_interaction.l_max,
        interaction_type=base_interaction.type,
    )

    if bases is not None:
        new_bases = []
        for interaction_list, basis_list in zip(interactions.interactions, bases):
            new_basis_list = []
            for interaction, basis in zip(interaction_list, basis_list):
                new_basis_list.append(
                    CustomBasis(
                        basis.pillars,
                        basis.phi_0,
                        basis.rho_mesh,
                        n_basis,
                        use_svd=False,
                        scale=False,
                        center=False,
                        subtract_phi0=False,
                        solver=basis.solver,
                    )
                )
                new_basis_list[-1].singular_values = basis.singular_values[:n_basis]
            new_bases.append(new_basis_list)
        emulator = ScatteringAmplitudeEmulator(
            interactions,
            new_bases,
            s_0=basis.solver.s_0,
            **SAE_kwargs,
        )
    else:
        emulator = ScatteringAmplitudeEmulator.from_train(
            interactions,
            theta_train,
            n_basis=n_basis,
            **SAE_kwargs,
        )

    return interactions, emulator


class CATPerformance:
    def __init__(
        self,
        benchmark_runner: Callable[[np.array], np.array],
        benchmark_inputs: list,
        benchmark_ground_truth: np.array,
        label: str = None,
        n_timing: int = 1,
    ):
        r"""
        Run benchmark_runner for each of benchmark_inputs, and compare the output
        to each of benchmark_data, returning performance metrics for each case

        Parameters:
            benchmark_runner : runs the benchmark, taking in a set of inputs in the form of an numpy
                array, and returning a set of outputs in another numpy array
            benchmark_inputs : list of inputs to benchmark runner, each formatted as an np.array
            benchmark_ground_truth : list corresponding to each input in benchmark runner, where each
                element is an np.array of same shape as output of benchmark_runner, encapsulating
                the 'exact', or expected, output for a given input
            label : name or identifier for this particular benchmark_runner
            n_timing : number of times to run benchmark_runner, the average of which is the time

        Returns:
            CATPerformance object encapsulating the mean squared error of each output of
            benchmark_runner compared to benchmark_ground_truth, as well as the elapsed computational
            time for each run of benchmark_runner

        Attributes:
            output_shape (tuple): shape of numpy array associated with output of benchmark_runner
            num_inputs (int): number of inputs passed to benchmark_runner
            runner_residuals (np.array): difference between benchmark_runner output and
                benchmark_ground_truth in each input, for each element of output
            rel_err (np.array) : same as runner_residuals but relative to benchmark_ground_truth
            times (np.array) : elapsed computational time for call to benchmark_runner for each
                of benchmark_runner
            median_rel_err : median of rel_err across output space
        """

        self.label = label
        self.output_shape = benchmark_ground_truth[0].shape
        self.num_inputs = len(benchmark_inputs)
        all_output_shape = (self.num_inputs,) + self.output_shape
        self.runner_residuals = np.zeros(
            all_output_shape, dtype=benchmark_ground_truth[0].dtype
        )
        self.runner_output = np.zeros_like(self.runner_residuals)
        self.rel_err = np.zeros(all_output_shape, dtype=np.double)
        self.times = np.zeros(self.num_inputs)
        for i in tqdm(range(self.num_inputs)):
            for j in range(int(n_timing)):
                st = perf_counter()
                predicted = benchmark_runner(benchmark_inputs[i])
                et = perf_counter()
                self.times[i] += et - st
            self.times[i] /= n_timing
            self.runner_output[i, ...] = predicted
            self.runner_residuals[i, ...] = benchmark_ground_truth[i] - predicted
            self.rel_err[i, ...] = np.sqrt(
                np.real(
                    self.runner_residuals[i, ...] * self.runner_residuals[i, ...].conj()
                )
            ) / np.sqrt(
                np.real(benchmark_ground_truth[i] * benchmark_ground_truth[i].conj())
            )

        # take median along all axes but 0
        axes = [i + 1 for i in range(len(self.output_shape))]
        self.median_rel_err = np.median(self.rel_err, axis=axes)


def CAT_plot(data_sets: list, labels=None, border_styles=None):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=400)

    # plt.rc("xtick")
    # plt.rc("ytick")

    custom_lines = []
    col = iter(colors)

    xlims = [np.inf, -np.inf]
    ylims = [np.inf, -np.inf]
    legend_locs = iter(["upper right", "lower right", "upper left", "lower left"])

    def reset_lims(x, y):
        xlims[0] = min(np.min(x), xlims[0])
        xlims[1] = max(np.max(x), xlims[1])
        ylims[0] = min(np.min(y), ylims[0])
        ylims[1] = max(np.max(y), ylims[1])

    def make_plot(data_set: CATPerformance, color, border_style="dashed"):
        custom_lines.append(
            Line2D(
                [],
                [],
                color=color,
                marker="o",
                linestyle=border_style,
                markersize=10,
                label=data_set.label,
            )
        )

        level_sns = 0.01

        x = data_set.times
        y = data_set.median_rel_err * 100

        reset_lims(x, y)

        sns.kdeplot(
            x=x,
            y=y,
            levels=[level_sns, 1],
            color=color,
            log_scale=[True, True],
            fill=True,
            alpha=0.6,
        )

        sns.kdeplot(
            x=x,
            y=y,
            levels=[level_sns],
            color=color,
            log_scale=[True, True],
            linewidths=3,
            linestyles=border_style,
        )
        ax.scatter(x, y, s=5, color=color)

    if isinstance(data_sets[0], list):
        for i, sub_list in enumerate(data_sets):
            custom_lines = []
            for j, data_set in enumerate(sub_list):
                if border_styles is not None:
                    border_style = border_styles[i]
                else:
                    border_style = "dashed"
                make_plot(data_set, next(col), border_style=border_style)
            label = labels[i] if labels is not None else None
            loc = next(legend_locs, "best")
            l = ax.legend(
                handles=custom_lines,
                frameon=True,
                edgecolor="black",
                title=label,
                loc=loc,
            )
            ax.add_artist(l)
    else:
        for i, data_set in enumerate(data_sets):
            make_plot(data_set, next(col))

        ax.legend(
            handles=custom_lines, frameon=True, edgecolor="black", loc="lower right"
        )

    xlims[0] *= 0.8
    ylims[0] *= 0.5
    xlims[1] *= 1.2
    ylims[1] *= 1.5
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("time per sample (s)")
    ax.set_ylabel("median relative error [%]")
    plt.tight_layout()

    return fig, ax


# fancy plotting stuff from https://stackoverflow.com/a/53586826
def multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$-\frac{%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex="\pi"):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )


def plot_phase_shifts(fig, ax1, ax2, deltas, shift=0, color=None, xlabel=True):
    r"""
    Plots the spin1/2-spin0 coupled phase shifts, the imaginary parts
    on ax2 and the real parts on ax1

    """
    l = np.array(list(range(len(deltas))))
    deltas_plus = np.array([d[0] for d in deltas])
    deltas_minus = np.array([d[1] for d in deltas[1:]])
    if color is not None:
        p = ax1.plot(
            l + shift,
            deltas_plus.real,
            color=color,
            marker=".",
            linestyle="solid",
            alpha=0.5,
        )[0]
    else:
        p = ax1.plot(
            l + shift,
            deltas_plus.real,
            color=color,
            marker=".",
            linestyle="solid",
            alpha=0.5,
        )[0]
    ax1.plot(
        l[1:] + shift,
        deltas_minus.real,
        marker="x",
        color=p.get_color(),
        linestyle="dotted",
        alpha=0.5,
    )
    ax2.plot(
        l + shift,
        deltas_plus.imag,
        marker=".",
        color=p.get_color(),
        linestyle="solid",
        alpha=0.5,
    )
    ax2.plot(
        l[1:] + shift,
        deltas_minus.imag,
        marker="x",
        color=p.get_color(),
        linestyle="dotted",
        alpha=0.5,
    )

    ax1.set_ylabel(r"$\mathfrak{Re}\,\delta$ [radians]")
    ax1.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 6))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator=6)))
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel(r"$\mathfrak{Im}\,\delta$ [radians]")
    ax2.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 6))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator=6)))

    if xlabel:
        ax1.set_xlabel(r"$\ell$ [$\hbar$]")
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.set_xlabel(r"$\ell$ [$\hbar$]")
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    styles = [
        Line2D(
            [0],
            [0],
            color="k",
            linestyle="solid",
            marker=".",
            label=r"$\delta_+$",
        ),
        Line2D(
            [0],
            [0],
            color="k",
            linestyle="dotted",
            marker="x",
            label=r"$\delta_-$",
        ),
    ]
    return styles, p.get_color()


def compare_phase_shifts(data_sets: list, labels: list, fig, ax1, ax2):
    color_legend = []

    for i, (deltas, label) in enumerate(zip(data_sets, labels)):
        # plot each one with small shift for overlap
        styles, c = plot_phase_shifts(fig, ax1, ax2, deltas, shift=i * 0.03)
        color_legend.append(
            Line2D([0], [0], color=c, linestyle="-", label=label),
        )
    leg1 = ax2.legend(handles=color_legend,)
    ax1.legend(handles=styles,)
    ax2.add_artist(leg1)

    return fig, ax1, ax2


def compare_phase_shifts_err(
    delta1,
    delta2,
    label1,
    label2,
    fig,
    ax1,
    ax2,
    ax3,
    ax4,
    small_label1=None,
    small_label2=None,
):
    fig, ax1, ax2 = compare_phase_shifts(
        [delta1, delta2],
        [label1, label2],
        fig,
        ax1,
        ax2,
    )

    diffs = []
    for l in range(len(delta1)):
        diff = []
        for j in range(len(delta1[l])):
            residual = delta1[l][j] - delta2[l][j]
            diff.append(np.fabs(residual.real) + 1j * np.fabs(residual.imag))
        diffs.append(diff)

    plot_phase_shifts(fig, ax3, ax4, diffs, color="k", xlabel=None)
    if small_label1 is None:
        small_label1 = label1
    if small_label2 is None:
        small_label2 = label2
    ax3.set_ylabel(
        r"$\mathfrak{Re}\,\left| \delta_{%s} - \delta_{%s} \right|$"
        % (small_label1, small_label2)
    )
    ax4.set_ylabel(
        r"$\mathfrak{Im}\,\left| \delta_{%s} - \delta_{%s} \right|$ "
        % (small_label1, small_label2)
    )

    ax3.set_yscale("log")
    ax4.set_yscale("log")
    ax3.relim()
    ax3.autoscale()
    ax4.relim()
    ax4.autoscale()

    ylim3 = ax3.get_ylim()
    ylim4 = ax4.get_ylim()
    ylim = [min(ylim3[0], ylim4[0]), max(ylim3[1], ylim4[1])]
    ax3.set_ylim(ylim)
    ax4.set_ylim(ylim)

    plt.tight_layout()


def plot_wavefunctions(
    s_mesh,
    wavefunctions,
    fig,
    ax1,
    ax2,
    linestyle="-",
    col_iter=iter(colors),
    partial_wave_offset: int = 0,
):
    r"""
    Plots the spin1/2-spin0 coupled reduced radial wavefunctions, the imaginary parts
    on ax2 and the real parts on ax1
    """

    legend_colors = []
    lwaves = [
        "s",
        "p",
        "d",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "q",
        "r",
        "t",
        "u",
        "v",
    ]
    lwaves_iter = iter(lwaves[partial_wave_offset:])

    for l, u_list in enumerate(wavefunctions):
        lsf = next(lwaves_iter)
        for i, u in enumerate(u_list):
            caption = r"$%s_{1/2}$" % lsf if i % 2 == 0 else r"$%s_{3/2}$" % lsf
            c = next(col_iter)
            ax1.plot(s_mesh, u.real, linestyle=linestyle, color=c, alpha=0.5)
            ax2.plot(s_mesh, u.imag, linestyle=linestyle, color=c, alpha=0.5)
            legend_colors.append(
                Line2D([0], [0], color=c, linestyle=linestyle, alpha=0.8, label=caption)
            )
            if l == 0:
                legend_colors.append(
                    Line2D([0], [0], color="w", linestyle=None, label=None)
                )

    fig.legend(
        handles=legend_colors,
        ncol=len(wavefunctions),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        bbox_transform=plt.gcf().transFigure,
    )

    Npi = int(np.max(s_mesh) / np.pi)
    denom = 6 // Npi
    if denom == 0:
        denom = 1

    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / denom))
    ax1.xaxis.set_major_formatter(
        plt.FuncFormatter(multiple_formatter(denominator=denom))
    )
    ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / denom))
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(multiple_formatter(denominator=denom))
    )

    ax1.set_xlabel(r"$s = kr$ [dimensionless]")
    ax1.set_ylabel(r"$\mathfrak{Re} \, u_{lj}(s)$ [a.u.]")
    ax2.set_xlabel(r"$s = kr$ [dimensionless]")
    ax2.set_ylabel(r"$\mathfrak{Im} \, u_{lj}(s)$ [a.u.]")
    plt.tight_layout()

    return fig, ax1, ax2


def compare_partial_waves(s_mesh, data_sets, labels, fig, ax1, ax2, l_start=None):
    linestyles = ["solid", "dotted", "dashed", "dotdashed"]
    assert len(data_sets) <= 4
    style_legend = []

    if l_start is None:
        l_start = 0

    for i, (data_set, label) in enumerate(zip(data_sets, labels)):
        col_iter = iter(colors)
        fig, ax1, ax2 = plot_wavefunctions(
            s_mesh,
            data_set,
            fig,
            ax1,
            ax2,
            linestyle=linestyles[i],
            col_iter=col_iter,
            partial_wave_offset=l_start,
        )
        style_legend.append(
            Line2D([0], [0], color="k", linestyle=linestyles[i], label=label),
        )
    leg1 = ax1.legend(handles=style_legend, loc="lower right")
    ax1.add_artist(leg1)

    return fig, ax1, ax2
