SubspaceKMeans
==============

Mautz, Dominik, et al. "Towards an Optimal Subspace for K-Means." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017. ([link](https://dl.acm.org/doi/10.1145/3097983.3097989))

Original implementation of above article.

This package provides `SubspaceKMeans` class which implements the above algorithm and act like the `KMeans` of scikit-learn (sklearn.cluster.KMeans).

## Install

`> pip install "git+https://github.com/tetutaro/subspacekmeans"`

## Very simple usage

```
from subspacekmeans import SubspaceKMeans

subspace_km = SubspaceKMeans(n_clusters=8)
predicted = subspace_km.fit_predict(data)
transformed = subspace_km.transform(data)
```

## Notices

- the `subspacekmeans` package is now based on `scikit-learn==0.24.1`
- the `SubspaceKMeans` class does not support sparse matrix as input data
- the `subspaceKMeans` class supports only `elkan` algorithm
    - It was just a hassle to support both (`elkan` and `lloyd`) algorithms. I think it is no problem to support `lloyd` algorithm.

## Detailed usage

see [usage.ipynb](usage.ipynb)

## Sample

![](cluster_space.png)
![](noise_space.png)
