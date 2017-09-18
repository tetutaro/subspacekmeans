#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""subspace k-Means clustering"""
import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _validate_center_shape
from sklearn.cluster.k_means_ import _tolerance
from sklearn.cluster.k_means_ import _labels_inertia
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import as_float_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.cluster import _k_means


def subspace_k_means(
    X,
    n_clusters,
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=1e-4,
    tol_eig=-1e-10,
    verbose=False,
    random_state=None,
    copy_x=True,
    n_jobs=1,
    return_n_iter=False
):
    if sp.issparse(X):
        raise ValueError("SubspaceKMeans does not support sparse matrix")
    if n_init <= 0:
        raise ValueError(
            "Invalid number of initializations."
            " n_init=%d must be bigger than zero." % n_init
        )
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError(
            'Number of iterations should be a positive number,'
            ' got %d instead' % max_iter
        )

    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    # Validate init array
    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # subtract of mean of x for more accurate distance computations
    X_mean = X.mean(axis=0)
    # The copy was already done above
    X -= X_mean

    if hasattr(init, '__array__'):
        init -= X_mean

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = subspace_kmeans_single(
                X,
                n_clusters,
                init=init,
                max_iter=max_iter,
                tol=tol,
                tol_eig=tol_eig,
                verbose=verbose,
                x_squared_norms=x_squared_norms,
                random_state=seeds[it]
            )
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(subspace_kmeans_single)(
                X,
                n_clusters,
                init=init,
                max_iter=max_iter,
                tol=tol,
                tol_eig=tol_eig,
                verbose=verbose,
                x_squared_norms=x_squared_norms,
                # Change seed to ensure variety
                random_state=seed
            ) for seed in seeds
        )
        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if not copy_x:
        X += X_mean
    best_centers += X_mean

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


def subspace_kmeans_single(
    X,
    n_clusters,
    init='k-means++',
    max_iter=300,
    tol=1e-4,
    tol_eig=-1e-10,
    verbose=False,
    x_squared_norms=None,
    random_state=None
):
    random_state = check_random_state(random_state)

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids(
        X,
        n_clusters,
        init,
        random_state=random_state,
        x_squared_norms=x_squared_norms
    )
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # === Beginning of original implementation of initialization ===

    # Dimensionality of original space
    d = X.shape[1]

    # Set initial V as QR-decomposed Q of random matrix
    rand_vals = random_state.random_sample(d ** 2).reshape(d, d)
    V, _ = np.linalg.qr(rand_vals, mode='complete')

    # Set initial m as d/2
    m = d // 2

    # Scatter matrix of the dataset in the original space
    S_D = np.dot(X.T, X)

    # Projection onto the first m attributes
    P_C = np.eye(m, M=d).T

    # === End of original implementation of initialization ===

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()

        # === Beginning of original implementation of E-step of EM ===

        X_C = np.dot(np.dot(X, V), P_C)
        mu_C = np.dot(np.dot(centers, V), P_C)
        labels, _ = pairwise_distances_argmin_min(
            X=X_C,
            Y=mu_C,
            metric='euclidean',
            metric_kwargs={'squared': True}
        )
        labels = labels.astype(np.int32)

        # === End of original implementation of E-step of EM ===

        # computation of the means is also called the M-step of EM
        centers = _k_means._centers_dense(X, labels, n_clusters, distances)

        # === Beginning of original implementation of M-step of EM ===

        S = np.zeros((d, d))
        for i in range(n_clusters):
            X_i = X[:][labels == i] - centers[:][i]
            S += np.dot(X_i.T, X_i)
        Sigma = S - S_D
        evals, evecs = np.linalg.eigh(Sigma)
        idx = np.argsort(evals)[::1]
        V = evecs[:, idx]
        m = len(np.where(evals < tol_eig)[0])
        if m == 0:
            raise ValueError(
                'Dimensionality of clustered space is 0. '
                'The dataset is better explained by a single cluster.'
            )
        P_C = np.eye(m, M=d).T
        inertia = 0.0
        for i in range(n_clusters):
            inertia += row_norms(
                X[:][labels == i] - centers[:][i],
                squared=True
            ).sum()

        # === End of original implementation of M-step of EM ===

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(X, x_squared_norms, best_centers,
                            precompute_distances=False,
                            distances=distances)

    return best_labels, best_inertia, best_centers, i + 1


class SubspaceKMeans(KMeans):
    """Subspace k-Means clustering

    Read more in
    Mautz, Dominik, et al.
    "Towards an Optimal Subspace for K-Means."
    Proceedings of the 23rd ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining. ACM, 2017.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    tol_eig : float, default: -1e-10
        Absolute tolerance with regards to eigenvalue of V to assume as 0

    verbose : int, default 0
        Verbosity mode.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    m_ : integer
        Dimensionality of the clusterd space

    V_ : float ndarray with shape (n_features, n_features)
        orthonormal matrix of a rigid transformation

    feature_importances_ : array of shape = [n_features]
        The transformed feature importances
        (the smaller, the more important the feature)
        (negative value (< tol_eig): feature of clustered space)
        (positive value (>= tol_eig): feature fo noise space).

    n_iter_ : int
        Number of iterations corresponding to the best results.
    """
    def __init__(
        self,
        n_clusters=8,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        tol_eig=-1e-10,
        verbose=0,
        random_state=None,
        copy_x=True,
        n_jobs=1
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.tol_eig = tol_eig
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        return

    def fit(self, X, y=None):
        """Compute subspace k-Means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored

        """
        if sp.issparse(X):
            raise ValueError("SubspaceKMeans does not support sparse matrix")
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            subspace_k_means(
                X,
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                tol_eig=self.tol_eig,
                verbose=self.verbose,
                random_state=random_state,
                copy_x=self.copy_x,
                n_jobs=self.n_jobs,
                return_n_iter=True
            )

        # === Beginning of original implementation of additional info ===
        d = X.shape[1]
        S_D = np.dot(X.T, X)
        S = np.zeros((d, d))
        for i in range(self.n_clusters):
            X_i = X[:][self.labels_ == i] - self.cluster_centers_[:][i]
            S += np.dot(X_i.T, X_i)
        Sigma = S - S_D
        self.feature_importances_, self.V_ = np.linalg.eigh(Sigma)
        self.m_ = len(np.where(self.feature_importances_ < self.tol_eig)[0])
        # === End of original implementation of additional info ===

        return self

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse=False, dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError(
                "Incorrect number of features. "
                "Got %d features, expected %d" %
                (n_features, expected_n_features)
            )
        return as_float_array(X, copy=self.copy_x)

    def _transform(self, X):
        return np.dot(X, self.V_)

    def inverse_transform(self, X, copy=None):
        return self.transform(X)
