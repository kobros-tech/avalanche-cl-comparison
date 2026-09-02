"""
A small synthetic class-incremental benchmark that requires no downloads.

Each class is a 2D-cluster-like Gaussian blob embedded in `input_dim`
dimensions (default 20), which keeps a tiny MLP fast to train while still
giving EWC/Replay/SkillMemory/etc. genuinely separable-but-nontrivial
classes to differentiate.

Used by every notebook in this project so results are reproducible without
network access. Swap in `avalanche.benchmarks.SplitMNIST(...)` (or any other
classic Avalanche benchmark) instead of `make_synthetic_benchmark(...)`
wherever you actually have dataset downloads available -- the rest of the
notebooks are benchmark-agnostic.
"""
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from avalanche.benchmarks import nc_benchmark, dataset_benchmark


def _make_class_centers(n_classes: int, input_dim: int, seed: int, class_sep: float = 3.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_classes, input_dim, generator=g) * class_sep


def _sample_from_centers(centers: torch.Tensor, samples_per_class: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    n_classes, input_dim = centers.shape
    xs, ys = [], []
    for c in range(n_classes):
        x = centers[c] + torch.randn(samples_per_class, input_dim, generator=g)
        xs.append(x)
        ys.append(torch.full((samples_per_class,), c, dtype=torch.long))
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    perm = torch.randperm(X.size(0), generator=g)
    return X[perm], Y[perm]


class _TargetsTensorDataset(TensorDataset):
    """Plain TensorDataset + a `.targets` attribute, which Avalanche's
    nc_benchmark needs to do its class-incremental split."""
    def __init__(self, x, y):
        super().__init__(x, y)
        self.targets = y.tolist()


def make_synthetic_benchmark(
    n_experiences: int = 5,
    n_classes: int = 10,
    input_dim: int = 20,
    train_samples_per_class: int = 300,
    test_samples_per_class: int = 100,
    seed: int = 0,
):
    """Returns (benchmark, input_dim, n_classes)."""
    assert n_classes % n_experiences == 0, "n_classes must be divisible by n_experiences"

    centers = _make_class_centers(n_classes, input_dim, seed, class_sep=3.0)
    Xtr, Ytr = _sample_from_centers(centers, train_samples_per_class, seed=seed + 1)
    Xte, Yte = _sample_from_centers(centers, test_samples_per_class, seed=seed + 2)

    train_ds = _TargetsTensorDataset(Xtr, Ytr)
    test_ds = _TargetsTensorDataset(Xte, Yte)

    benchmark = nc_benchmark(
        train_dataset=train_ds,
        test_dataset=test_ds,
        n_experiences=n_experiences,
        task_labels=False,
        shuffle=True,
        seed=seed,
    )
    return benchmark, input_dim, n_classes


def make_transfer_demo_benchmark(
    input_dim: int = 20,
    train_samples_per_class: int = 150,
    test_samples_per_class: int = 60,
    seed: int = 0,
    shift_scale: float = 1.6,
):
    """A hand-built 5-experience stream designed to exercise all three
    Skill Memory actions with a known "correct" answer, matching Section
    12's Level 2 (oracle transfer) experiment:

        exp 0: classes {0, 1}                 -> expect SCRATCH (memory empty)
        exp 1: classes {2, 3}, unrelated       -> expect SCRATCH (no compatible skill)
        exp 2: classes {0, 1}, centers shifted -> expect CLONE   (related but not identical
                                                                    -> needs a little adaptation)
        exp 3: classes {0, 1}, identical dist. -> expect REUSE   (exact same task as exp 0)
        exp 4: classes {4, 5}, unrelated       -> expect SCRATCH (still no compatible skill)

    Returns (train_list, test_list, classes_list, input_dim, n_classes, ground_truth).
    `train_list`/`test_list` are plain lists of TensorDatasets (NOT wrapped
    in an Avalanche NCScenario, since NCScenario assumes a disjoint
    class-incremental partition and this benchmark deliberately repeats
    classes across experiences).
    """
    # Deliberately harder-than-trivial task (lower class separation, higher
    # noise) so that a fresh model needs several epochs to fit well -- this
    # is what lets REUSE/CLONE show a genuine advantage over a
    # probe-budget-limited SCRATCH run within these notebooks.
    n_classes = 6
    centers = _make_class_centers(n_classes, input_dim, seed, class_sep=1.6)

    # A small, consistent perturbation applied to classes {0,1} for exp 2 --
    # "related but not identical" (same rough region of input space).
    g = torch.Generator().manual_seed(seed + 42)
    shift = torch.randn(2, input_dim, generator=g) * shift_scale
    centers_shifted = centers.clone()
    centers_shifted[0:2] += shift

    def build(class_ids, ctrs, n_per_class, seed):
        sub_centers = ctrs[class_ids]
        X, Yrel = _sample_from_centers(sub_centers, n_per_class, seed=seed)
        # remap the *relative* 0..len(class_ids)-1 labels back to the
        # original global class ids.
        Y = torch.tensor(class_ids, dtype=torch.long)[Yrel]
        return _TargetsTensorDataset(X, Y)

    specs = [
        ("exp0_base_01", [0, 1], centers, "SCRATCH"),
        ("exp1_unrelated_23", [2, 3], centers, "SCRATCH"),
        ("exp2_shifted_01", [0, 1], centers_shifted, "CLONE"),
        ("exp3_identical_01", [0, 1], centers, "REUSE"),
        ("exp4_unrelated_45", [4, 5], centers, "SCRATCH"),
    ]

    train_list, test_list, classes_list, ground_truth = [], [], [], []
    for i, (name, class_ids, ctrs, expected) in enumerate(specs):
        train_list.append(build(class_ids, ctrs, train_samples_per_class, seed=seed + 100 + i))
        test_list.append(build(class_ids, ctrs, test_samples_per_class, seed=seed + 200 + i))
        classes_list.append(class_ids)
        ground_truth.append(expected)

    return train_list, test_list, classes_list, input_dim, n_classes, ground_truth
