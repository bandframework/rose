import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from numba import njit
from scipy.spatial import kdtree
from mpmath import coulombf

from .interaction import InteractionSpace
from .scattering_amplitude_emulator import ScatteringAmplitudeEmulator

#TODO multithreaded training

class ActiveSubspaceQuilt:
    """Each patch on the quilt approximates a tangent space to the full operator via a Reduced Basis emulator built from training points in the neighborhood. These neighborhoods are constructed not on the parameter space, but the active subspace as approxximated by the training data"""

    def __init__(
        self,
        interactions: InteractionSpace,
        solver: ScatteringAmplitudeEmulator,
        s_mesh: np.array,
        s_0: float,
        bounds,
        train,
        forward_pspace_transform,
        backward_pspace_transform,
        frozen_mask,
        neighborhood_size=100,
        tangent_fraction=0.1,
    ):
        self.interactions = interactions
        self.solver = solver
        self.s_mesh = s_mesh
        self.s_0 = s_0
        self.domain = np.array([self.s_mesh[0], self.s_mesh[-1]])
        self.l_max = solver.l_max
        self.neighborhood_size = neighborhood_size
        self.frozen_mask = frozen_mask.copy()
        self.unfrozen_mask = np.logical_not(self.frozen_mask)
        self.forward_pspace_transform = forward_pspace_transform
        self.backward_pspace_transform = backward_pspace_transform

        self.bounds = np.vstack(
            [
                bounds[:, 0],
                bounds[:, 1],
            ]
        ).T

        self.ndims_total = len(self.frozen_mask)
        self.ndims_active = self.ndims_total - np.sum(self.frozen_mask)

        # calculate free solutions
        self.free_solns = np.zeros((self.l_max + 1, s_mesh.size), dtype=np.complex128)
        for l in range(0, self.l_max + 1):
            self.free_solns[l, :] = np.array(
                [coulombf(l, 0, s) for s in s_mesh], dtype=np.complex128
            )
            self.free_solns[l, :] /= np.trapz(np.absolute(free_solns[l, :]), s_mesh)

        self.update_train(train)
        self.update_tangents()

    def get_tangent_space(self, sample):
        dist, idx = self.tangent_tree.query(self.to_AS(sample))
        tangent_idx = self.tangent_idxs[idx]
        tangent_point = self.train[tangent_idx, :]
        dists, neighborhood = self.active_subspace_kdtree.query(
            self.train_as[tangent_idx, :]
        )
        return dist, idx, (dists, neighborhood)

    def get_local_emulator(self, sample):
        dist, idx = self.tangent_tree.query(self.to_AS(sample))
        return self.emulators[idx]

    def update_tangents(self):
        ntangents = int(np.ceil(self.tangent_fraction * self.train.shape[0]))
        ntangents += 1
        self.tangent_idxs = np.random.choice(
            np.arange(0, self.train.shape[0], 1, dtype=int), ntangents, replace=False
        )
        self.tangent_tree = kdtree.KDTree(self.train_as[self.tangent_idxs, :])
        self.emulators = []
        for t in tangent_idxs:
            ds, neighbors = self.active_subspace_kdtree.query(
                self.train_as[t, :], k=self.neighborhood_size
            )
            interactions = rose.koning_delaroche.EnergizedKoningDelaroche(
                training_info=self.train[neighbors, :],
                explicit_training=True,
                n_basis=15,
                l_max=self.l_max,
                rho_mesh=self.s_mesh,
            )
            bases = self.make_bases(neighbors, interactions)
            self.emulators.append(
                (
                    interactions,
                    rose.ScatteringAmplitudeEmulator(
                        interaction_space=interactions,
                        bases=bases,
                        angles=self.angles,
                        s_0=self.s_0,
                        l_max=self.l_max,
                    ),
                )
            )

    def make_bases(self, neighbors, interactions, nbasis=10):
        bases = []
        for l in range(l_max + 1):
            bup = rose.CustomBasis(
                self.hf_solns[neighbors, l, :].T,
                self.free_solns[l, :],
                self.s_mesh,
                nbasis,
                solver=rose.SchroedingerEquation(
                    self.interactions.interactions[l][0],
                    s_0=self.s_0,
                    domain=self.domain,
                ),
                use_svd=True,
                scale=False,
                subtract_phi0=False,
            )
            if l == 0:
                bases.append([bup])
            else:
                bdown = rose.CustomBasis(
                    self.hf_solns[neighbors, self.l_max + l, :].T,
                    self.free_solns[l, :],
                    self.s_mesh,
                    nbasis,
                    solver=rose.SchroedingerEquation(
                        interactions.interactions[l][1],
                        s_0=s_0,
                        domain=domain,
                    ),
                    use_svd=True,
                    scale=False,
                    subtract_phi0=False,
                )
                bases.append([bup, bdown])
        return bases

    def update_train(self, new_train):
        # update train
        if self.train.empty():
            self.train = new_train
        else:
            self.train = np.hstack([self.train, new_train])

        # update bounds
        new_bounds = np.vstack(
            [
                np.min(train, axis=0),
                np.max(train, axis=0),
            ]
        ).T
        self.bounds[self.bounds[:, 0] > new_bounds[:, 0], 0] = new_bounds[:, 0]
        self.bounds[self.bounds[:, 1] < new_bounds[:, 1], 1] = new_bounds[:, 1]
        self.bounds_transformed = np.vstack(
            [
                self.forward_pspace_transform(bounds[:, 0]),
                self.forward_pspace_transform(bounds[:, 1]),
            ]
        ).T

        # update pre-processing to use new mean and bounds
        self.train_mean = self.forward_pspace_transform(np.mean(train, axis=0))
        self.train_pp = np.array([self.pre_process(sample) for sample in train])
        self.tree = kdtree.KDTree(self.train_pp)
        self.prepro_bounds = np.vstack(
            [self.pre_process(self.bounds[:, 0]), self.pre_process(self.bounds[:, 1])]
        ).T

        # calculate and store interaction terms
        new_interaction_terms = np.zeros(
            new_train.shape[0],
            2 * self.l_max + 1,
            self.s_mesh.shape[0],
            dtype=np.complex128,
        )
        for i, sample in enumerate(new_train):
            for l in range(0, self.l_max + 1):
                new_interaction_terms[i, l, :] = self.interactions.interactions[l][
                    0
                ].tilde(self.s_mesh, sample)
            for l in range(1, self.l_max + 1):
                new_interaction_terms[
                    i, self.l_max + l, :
                ] = self.interactions.interactions[l][1].tilde(self.s_mesh, sample)

        if self.train.empty():
            self.interaction_terms = new_interaction_terms
        else:
            self.interaction_terms = np.hstack(
                [self.interaction_terms, new_interaction_terms]
            )

        # calculate and store HF solutions
        new_hf_solns = np.zeros(
            new_train.shape[0],
            2 * self.l_max + 1,
            self.s_mesh.shape[0],
            dtype=np.complex128,
        )
        for i, sample in enumerate(new_train):
            solns = self.solver.exact_wave_functions(sample)
            for l in range(0, self.l_max + 1):
                new_hf_solns[i, l, :] = (
                    solns[l][0] / np.trapz(np.absolute(solns[l][0]), s_mesh)
                    - self.free_solns[l, :]
                )
            for l in range(1, self.l_max + 1):
                new_hf_solns[i, self.l_max + l, :] = (
                    solns[l][1] / np.trapz(np.absolute(solns[l][1]), s_mesh)
                    - self.free_solns[l, :]
                )

        if self.train.empty():
            self.hf_solns = new_hf_solns
        else:
            self.hf_solns = np.hstack([self.hf_solns, new_hf_solns])

        # rediscover subspace
        self.U, self.S = self.discover()
        self.Utrans = self.U @ np.diag(self.S)
        self.Uinv = np.linalg.inv(self.Utrans)
        self.train_as = np.array([self.to_AS(sample) for sample in train])
        self.active_subspace_kdtree = kdtree.KDTree(self.train_as)

    def discover(self):
        lcut = self.l_max
        k = self.neighborhood_size
        gradient_vector_samples = np.zeros(
            (
                self.train.shape[0] * (lcut * 4 - 1),
                self.train[:, self.unfrozen_mask].shape[1],
            ),
            dtype=np.double,
        )

        for i in range(self.train.shape[0]):
            sample = self.train[i, :]
            ds, idxs = self.tree.query(self.pre_process(sample), k=k)

            max_delta_idxs = np.zeros((lcut * 4 - 2), dtype=np.int32)
            max_func_deriv = np.zeros((lcut * 4 - 2), dtype=np.complex128)

            for d, j in zip(ds, idxs):
                # numerator is \int ds phi_i^dagger (U_j(s) - U_i(s)) \phi_j
                phi_j_up = self.hf_solns[j, : lcut + 1, 0, :]
                phi_j_down = self.hf_solns[j, lcut + 1 : 2 * lcut, 1, :]

                func_deriv_up = (
                    np.trapz(
                        phi_j_up
                        * (
                            self.interaction_terms[j, : lcut + 1, 0, :]
                            - self.interaction_terms[j, : lcut + 1, 0, :]
                        )
                        * phi_j_up,
                        self.s_mesh,
                        axis=1,
                    )
                    / d
                )
                func_deriv_down = (
                    np.trapz(
                        phi_j_down
                        * (
                            self.interaction_terms[j, lcut + 1 : 2 * lcut, 1, :]
                            - self.interaction_terms[j, lcut + 1 : 2 * lcut, 1, :]
                        )
                        * phi_j_down,
                        self.s_mesh,
                        axis=1,
                    )
                    / d
                )
                func_deriv = np.hstack(
                    [
                        func_deriv_down.real,
                        func_deriv_up.real,
                        func_deriv_down.imag,
                        func_deriv_up.imag,
                    ]
                )
                mask = np.absolute(max_func_deriv) < np.absolute(func_deriv)
                max_func_deriv[mask] = func_deriv[mask]
                max_delta_idxs[mask] = j

            gradient_vector_samples[i : i + (lcut * 4 - 2), :] = [
                (self.train_pp[x, unfrozen_mask] - self.train_pp[i, unfrozen_mask])
                * np.sign(y)
                / np.sqrt(
                    np.dot(
                        self.train_pp[x, unfrozen_mask]
                        - self.train_pp[i, unfrozen_mask],
                        self.train_pp[x, unfrozen_mask]
                        - self.train_pp[i, unfrozen_mask],
                    )
                )
                for x, y in zip(max_delta_idxs, max_func_deriv)
            ]

        U, S, Vh = np.linalg.svd(gradient_vector_samples.T, full_matrices=False)
        return U, S

    def pre_process(self, sample):
        """
        scale and center point on parameter space (wrt to training set)
        """
        x = (self.forward_pspace_transform(sample) - self.train_mean) / (
            self.bounds_transformed[:, 1] - self.bounds_transformed[:, 0]
        )
        x[self.frozen_mask] = 0
        return x

    def post_process(self, sample):
        """
        Un-scale and un-center point on scaled and centered parameter space
        """
        x = self.backward_pspace_transform(
            sample * (self.bounds_transformed[:, 1] - self.bounds_transformed[:, 0])
            + self.train_mean
        )
        x[self.frozen_mask] = self.bounds[self.frozen_mask, 0]
        return x

    def to_AS(self, sample):
        """
        Convert from parameter space to active subspace
        """
        return self.Utrans @ self.pre_process(sample)[self.unfrozen_mask]

    def from_AS(self, point):
        """
        Convert from active subspace to parameter space
        """
        y = np.zeros((self.ndims_total))
        y[self.unfrozen_mask] = self.Uinv @ point
        y[self.frozen_mask] = self.bounds[self.frozen_mask, 0]
        return self.post_process(y)

    def get_neighbors(self, sample, knn=1):
        """
        Get nearest neighboring training points on active subspace to sample
        """
        dists, idxs = self.active_subspace_kdtree.query(self.to_AS(sample), k=knn)
        return dists, idxs

    def get_bbox(self, neighborhood):
        """
        Return bounding box (in centered and scaled space) containing a training neighborhood
        """
        neighbors = self.train[neighborhood]
        size = neighbors.shape[0]
        bbox = np.tile(self.pre_process(neighbors[0, :]).T, (2, 1)).T
        for i in range(size):
            neighbor = self.pre_process(neighbors[i, :])
            mask_lower = neighbor <= bbox[:, 0]
            mask_upper = neighbor > bbox[:, 1]
            bbox[mask_lower, 0] = neighbor[mask_lower]
            bbox[mask_upper, 1] = neighbor[mask_upper]

        return bbox

    def get_active_bbox(self, neighborhood):
        """
        Return bounding box (in centered and scaled space) containing a training neighborhood
        """
        neighbors = self.train_as[neighborhood, :]
        size = neighbors.shape[0]
        bbox = np.tile(neighbors[0, :].T, (2, 1)).T
        for i in range(size):
            neighbor = neighbors[i, :]
            mask_lower = neighbor <= bbox[:, 0]
            mask_upper = neighbor > bbox[:, 1]
            bbox[mask_lower, 0] = neighbor[mask_lower]
            bbox[mask_upper, 1] = neighbor[mask_upper]

        return bbox

    def sample_from_active_neighborhood(
        self, tangent_point, neigborhood_size, nsamples, seed=None
    ):
        """
        Sample points uniformly in the active subspace, from the training neighborhood belonging
        to tangent_point
        """
        dists, neighborhood = self.get_neighbors(tangent_point, knn=neigborhood_size)
        bbox = self.get_active_bbox(neighborhood)
        return np.array(
            [
                self.from_AS(sample)
                for sample in rose.training.latin_hypercube_sample(nsamples, bbox, seed)
            ]
        )

    def sample_from_neighborhood(
        self, tangent_point, neigborhood_size, nsamples, seed=None
    ):
        """
        Sample points uniformly in the parameter space, from the training neighborhood belonging
        to tangent_point
        """
        dists, neighborhood = self.get_neighbors(tangent_point, knn=neigborhood_size)
        bbox = self.get_bbox(neighborhood)
        return np.array(
            [
                self.post_process(sample)
                for sample in rose.training.latin_hypercube_sample(nsamples, bbox, seed)
            ]
        )

    def sample_from_active_neighborhood_gaussian(
        self, tangent_point, neigborhood_size, nsamples, seed=None
    ):
        """
        Sample points from a multivariate normal in the active subspace, centered around tangent_point
        and with covariance determined by the training neighborhood
        """
        dists, neighborhood = self.get_neighbors(tangent_point, knn=neigborhood_size)
        mean = self.to_AS(tangent_point)
        cov = np.cov(self.train_as[neighborhood, :])
        return np.random.multivariate_normal(mean, cov, size=nsamples)

    def sample_from_neighborhood_gaussian(
        self, tangent_point, neigborhood_size, nsamples, seed=None
    ):
        """
        Sample points from a multivariate normal in the parameter space, centered around tangent_point
        and with covariance determined by the training neighborhood
        """
        dists, neighborhood = self.get_neighbors(tangent_point, knn=neigborhood_size)
        mean = self.pre_process(tangent_point)
        cov = np.cov(self.train_pp[neighborhood, :])
        return np.array(
            [
                self.post_process(sample)
                for sample in np.random.multivariate_normal(mean, cov, size=nsamples)
            ]
        )

    def sample(self, nsamples, seed=None):
        """
        Sample points in parameters space
        """
        return np.array(
            [
                self.post_process(sample)
                for sample in rose.training.latin_hypercube_sample(
                    nsamples, self.prepro_bounds, seed
                )
            ]
        )

    def save(self, fpath):
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(obj, fpath):
        with open(fpath, "rb") as f:
            asubs = pickle.load(f)
        return asubs
