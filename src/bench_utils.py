"""
Shared benchmark utilities for the notebook suite.

By default this builds a small SYNTHETIC class-incremental benchmark using
`avalanche.benchmarks.nc_benchmark`, so every notebook in this project runs
fully offline / without downloading MNIST, CIFAR, etc. This is deliberate:
it keeps the project runnable in restricted/offline environments and keeps
iteration fast while comparing many strategies.

If you have internet access to torchvision's dataset mirrors, swap
`make_synthetic_benchmark(...)` for e.g. `SplitMNIST(n_experiences=5)` from
`avalanche.benchmarks` -- every notebook only depends on the resulting
object having `.train_stream` / `.test_stream` / `.n_classes`, so nothing
else needs to change. See `make_split_mnist_benchmark()` below for the
one-line swap.
"""
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from avalanche.benchmarks import nc_benchmark


def make_synthetic_benchmark(
    n_classes: int = 10,
    n_experiences: int = 5,
    feature_dim: int = 64,
    n_per_class: int = 300,
    class_sep: float = 1.6,
    noise: float = 1.0,
    seed: int = 0,
):
    """A small, fast, fully-offline class-incremental benchmark.

    Each class is a Gaussian blob in `feature_dim`-D space; `class_sep`
    controls how far apart the blob centers are (bigger = easier task,
    smaller = harder / more class confusion, which is what makes transfer
    and forgetting visible). `nc_benchmark` then splits the classes into
    `n_experiences` disjoint groups, exactly like SplitMNIST/SplitCIFAR do
    for real image datasets.
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, feature_dim, generator=g) * class_sep

    X, Y = [], []
    for c in range(n_classes):
        x = centers[c] + torch.randn(n_per_class, feature_dim, generator=g) * noise
        X.append(x)
        Y.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(X)
    Y = torch.cat(Y)

    perm = torch.randperm(len(X), generator=g)
    X, Y = X[perm], Y[perm]

    n_train = int(0.8 * len(X))
    train_x, test_x = X[:n_train], X[n_train:]
    train_y, test_y = Y[:n_train], Y[n_train:]

    train_ds = TensorDataset(train_x, train_y)
    train_ds.targets = train_y.tolist()
    test_ds = TensorDataset(test_x, test_y)
    test_ds.targets = test_y.tolist()

    benchmark = nc_benchmark(
        train_ds, test_ds,
        n_experiences=n_experiences,
        task_labels=False,
        shuffle=True,
        seed=seed,
    )
    benchmark.n_classes = n_classes
    benchmark.feature_dim = feature_dim
    return benchmark


def make_split_mnist_benchmark(n_experiences: int = 5, seed: int = 0):
    """Drop-in replacement for `make_synthetic_benchmark` using real
    MNIST, IF you have network access to torchvision's dataset mirrors.
    Not used by default in this project because sandboxed/offline
    environments usually can't reach those mirrors."""
    from avalanche.benchmarks import SplitMNIST
    benchmark = SplitMNIST(n_experiences=n_experiences, seed=seed)
    benchmark.n_classes = 10
    benchmark.feature_dim = 28 * 28
    return benchmark
