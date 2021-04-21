#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Subspace k-Means clustering"""
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_fast import _inertia_dense
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning


def subspace_kmeans_single(
    X,
    sample_weight,
    centers_init,
    random_state,
    max_iter=300,
    tol=1e-4,
    tol_eig=-1e-10,
    verbose=False
):
    """A single run of Subspace k-Means

    Parameters are about the same as
    sklearn.cluster._kmeans._kmeans_single_elkan.
    But following parmeters are added.

    Parameters
    ----------
    random_state : RandomState instance
        Determines random number generation for centroid initialization.

    tol_eig: float, default: -1e-10
        Absolute tolerance with regards to eigenvalue of V to assume as 0.

    Returns are the same as
    sklearn.cluster._kmeans._kmeans_single_elkan.
    """
    n_samples = X.shape[0]
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()

    # internal variables for subspace k-means
    # Dimension of original space
    d = X.shape[1]
    # Set initial V as QR-decomposed Q of random matrix
    V, _ = np.linalg.qr(
        random_state.random_sample(d ** 2).reshape(d, d),
        mode='complete'
    )
    # Set initial m as d/2
    m = d // 2
    if m == 0:
        m = 1
    # Scatter matrix of the dataset in the original space
    S_D = np.dot(X.T, X)
    # Projection onto the first m attributes
    P_C = np.eye(m, M=d).T

    for i in range(max_iter):
        # E-step: update labels
        X_C = np.dot(np.dot(X, V), P_C)
        mu_C = np.dot(np.dot(centers, V), P_C)
        labels, _ = pairwise_distances_argmin_min(
            X=X_C,
            Y=mu_C,
            metric='euclidean',
            metric_kwargs={'squared': True}
        )
        labels = labels.astype(np.int32)
        # M-step: update centers
        centers_new = np.stack([
            X[labels == c, :].mean(axis=0) for c in range(n_clusters)
        ])
        centers, centers_new = centers_new, centers
        if verbose:
            inertia = _inertia_dense(X, sample_weight, centers, labels)
            print(f"Iteration {i}, inertia {inertia}")
        # update internal variables for subspace k-means
        S = np.zeros((d, d))
        for c in range(n_clusters):
            X_c = X[labels == c, :] - centers[c, :]
            S += np.dot(X_c.T, X_c)
        Sigma = S - S_D
        evals, evecs = np.linalg.eigh(Sigma)
        V = evecs[:, np.argsort(evals)]
        m = len(np.where(evals < tol_eig)[0])
        if m == 0:
            raise ValueError(
                'Dimensionality of clustered space is 0. '
                'The dataset is better explained by a single cluster.'
            )
        P_C = np.eye(m, M=d).T
        # check convergence
        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = ((centers - centers_new) ** 2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break
        labels_old[:] = labels

    inertia = _inertia_dense(X, sample_weight, centers, labels)
    ret = {
        'labels': labels,
        'inertia': inertia,
        'centers': centers,
        'n_iter': i + 1,
        'cluster_dims': m,
        'V': V,
    }
    return ret


class SubspaceKMeans(KMeans):
    """Subspace k-Means clustering

    Mautz, Dominik, et al.
    "Towards an Optimal Subspace for K-Means."
    Proceedings of the 23rd ACM SIGKDD International Conference
    on Knowledge Discovery and Data Mining. ACM, 2017.

    Parameters are about the same as KMeans()
    but following parameters are added.

    Parameters
    ----------
    tol_eig: float, default: -1e-10
        Absolute tolerance with regards to eigenvalue of V to assume as 0.

    Attributes are also about the same as KMeans()
    but following attributes are added.

    Attributes
    ----------
    cluster_dims_ : integer
        number of dimension of the clusterd space

    V_ : float ndarray with shape (n_features, n_features)
        The orthonormal matrix of a rigid transformation
    """
    def __init__(
        self,
        n_clusters=8,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        precompute_distances='deprecated',
        verbose=0,
        random_state=None,
        copy_x=True,
        n_jobs='deprecated',
        algorithm='auto',
        tol_eig=-1e-10,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            precompute_distances=precompute_distances,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            n_jobs=n_jobs,
            algorithm=algorithm
        )
        self.tol_eig = tol_eig
        return

    def fit(self, X, y=None, sample_weight=None):
        """Compute subspace k-Means clustering.

        Parameters
        ----------
        X : array-like (not sparse) matrix, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self
        """
        X = self._validate_data(
            X, dtype=[np.float64, np.float32],
            accept_sparse=False, accept_large_sparse=False,
            order='C', copy=self.copy_x
        )
        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=X.dtype
        )
        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype, copy=True, order='C')
            self._validate_center_shape(X, init)
        # subtract of mean of x for more accurate distance computations
        X_mean = X.mean(axis=0)
        X -= X_mean
        if hasattr(init, '__array__'):
            init -= X_mean
        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)
        # initialize results
        best_inertia = None
        # run
        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init,
                random_state=random_state
            )
            if self.verbose:
                print("Initialization complete")
            # === call function for subspace k-Means ===
            ret = subspace_kmeans_single(
                X=X,
                sample_weight=sample_weight,
                centers_init=centers_init,
                random_state=random_state,
                max_iter=self.max_iter,
                tol=self.tol,
                tol_eig=self.tol_eig,
                verbose=self.verbose
            )
            inertia = ret['inertia']
            if best_inertia is None or inertia < best_inertia:
                best_labels = ret['labels']
                best_centers = ret['centers']
                best_inertia = inertia
                best_n_iter = ret['n_iter']
                best_cluster_dims = ret['cluster_dims']
                best_V = ret['V']
        if not self.copy_x:
            X += X_mean
        best_centers += X_mean
        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning, stacklevel=2
            )
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.cluster_dims_ = best_cluster_dims
        self.V_ = best_V
        return self

    def _transform(self, X):
        return np.dot(X, self.V_)

    def predict(self, X, sample_weight=None):
        check_is_fitted(self)
        X = self._check_test_data(X)
        d = X.shape[1]
        P_C = np.eye(self.cluster_dims_, M=d).T
        X_C = np.dot(np.dot(X, self.V_), P_C)
        mu_C = np.dot(np.dot(self.cluster_centers_, self.V_), P_C)
        labels, _ = pairwise_distances_argmin_min(
            X=X_C,
            Y=mu_C,
            metric='euclidean',
            metric_kwargs={'squared': True}
        )
        labels = labels.astype(np.int32)
        return labels

    def score(self, X, y=None, sample_weight=None):
        check_is_fitted(self)
        X = self._check_test_data(X)
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=X.dtype
        )
        d = X.shape[1]
        P_C = np.eye(self.cluster_dims_, M=d).T
        X_C = np.dot(np.dot(X, self.V_), P_C)
        mu_C = np.dot(np.dot(self.cluster_centers_, self.V_), P_C)
        labels, _ = pairwise_distances_argmin_min(
            X=X_C,
            Y=mu_C,
            metric='euclidean',
            metric_kwargs={'squared': True}
        )
        labels = labels.astype(np.int32)
        inertia = _inertia_dense(
            X, sample_weight, self.cluster_centers_, labels
        )
        return -inertia
