r"""
Some helpful utilities for training an emulator
"""
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from time import perf_counter
from scipy.stats import qmc

from .basis import Basis
from .interaction_eim import InteractionEIM, InteractionEIMSpace
from .scattering_amplitude_emulator import ScatteringAmplitudeEmulator


def sample_params_LHC(
    N: int, central_vals: np.array, scale: float = 0.5, seed: int = None
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
    bounds = np.array(
        [
            central_vals - np.fabs(central_vals * scale),
            central_vals + np.fabs(central_vals * scale),
        ]
    ).T
    return qmc.scale(
        qmc.LatinHypercube(d=central_vals.size, seed=seed).random(N),
        bounds[:, 0],
        bounds[:, 1],
    )


def CAT_trainer_EIM(
    sae_config: tuple,
    base_interaction: InteractionEIMSpace,
    bases: list = None,
    theta_train: np.array = None,
    **SAE_kwargs,
):
    r"""
    build an EIM emulator to specification of sae_config, using base_interaction
    Parameters:
        sae_config :  (size of reduced basis, number of EIM terms)
        base_interaction :  interaction on which to train
        bases (optional) : if a full set of bases for each interaction has been solved
            for already, re-use basis.vectors rather than re-calculating them
        theta_train (optional) : is bases is not provided, simply pass in the
            training samples and re-train the emulator
        SAE_kwargs : passed to ScatteringAmplitudeEmulator
    """

    (n_basis, n_EIM) = sae_config

    interactions = InteractionEIMSpace(
        base_interaction.coordinate_space_potential,
        base_interaction.n_theta,
        base_interaction.mu,
        base_interaction.energy,
        base_interaction.training_info,
        l_max=base_interaction.l_max,
        Z_1=base_interaction.Z_1,
        Z_2=base_interaction.Z_2,
        R_C=base_interaction.R_C,
        is_complex=base_interaction.is_complex,
        spin_orbit_potential=base_interaction.spin_orbit_potential,
        explicit_training=base_interaction.explicit_training,
        n_train=base_interaction.n_train,
        rho_mesh=base_interaction.rho_mesh,
        n_basis=n_EIM,
    )

    if theta_train is None:
        assert bases is not None
        new_bases = []
        for interaction_list, basis_list in zip(interactions.interactions, bases):
            basis_list = []
            for interaction, basis in zip(interaction_list, basis_list):
                # add back free solution to get HF solutions
                solutions = (basis.pillars.T + basis.phi_0).T
                basis_list.append(
                    CustomBasis(
                        solutions[:, :n_basis],
                        basis.phi_0,
                        basis.rho_mesh,
                        n_basis,
                        interaction.ell,
                        use_svd=False,
                    )
                )
            new_bases.append(basis_list)
        emulator = ScatteringAmplitudeEmulator(interaction, new_bases, **SAE_kwargs)
    else:
        emulator = ScatteringAmplitudeEmulator.from_train(
            interaction,
            training_samples,
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

        self.output_shape = benchmark_ground_truth[0].shape
        self.num_inputs = len(benchmark_inputs)
        all_output_shape = (self.num_inputs,) + self.output_shape
        self.runner_residuals = np.zeros(all_output_shape)
        self.rel_err = np.zeros(all_output_shape)
        self.times = np.zeros(self.num_inputs)
        for i in range(self.num_inputs):
            st = perf_counter()
            predicted = benchmark_runner(benchmark_inputs[i])
            et = perf_counter()
            self.runner_residuals[i, ...] = benchmark_ground_truth[i] - predicted
            self.rel_err[i, ...] = (
                np.fabs(self.runner_residuals[i, ...]) / benchmark_ground_truth[i]
            )
            self.times[i] = et - st

        self.median_rel_err = np.median(self.rel_err, axis=0)


def CAT_plot(data_sets: list):
    def make_plot(
        data_set: CATPerformance, color, runner_label_title="[$n_\phi$, $n_U$]"
    ):
        custom_lines = [
            Line2D(
                [],
                [],
                color="w",
                marker="X",
                linestyle="None",
                markersize=20,
                label=runner_label_title,
            ),
            Line2D(
                [],
                [],
                color=color,
                marker="o",
                linestyle="None",
                markersize=20,
                label=data_set.label,
            ),
        ]

        level_sns = 0.001

        x = data_set.times
        y = data_set.rel_err * 100

        sns.kdeplot(
            x=x,
            y=y,
            levels=[level_sns, 1],
            color=colors[i],
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
            linestyles="dashed",
        )
        ax.scatter(x, y, s=5, color=color)
        ax.legend(handles=custom_lines, frameon=True, edgecolor="black")

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

    if isinstance(data_sets[0], list):
        for i, data_set in enumerate(data_sets):
            make_plot(data_set, colors[i])
    else:
        make_plot(data_sets)

    fig, ax = plt.subplots(figsize=(15, 7), dpi=400)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("time per sample (s)")
    ax.set_ylabel("median relative error [%]")

    plt.rc("xtick")
    plt.rc("ytick")

    plt.tight_layout()

    return fig, ax
