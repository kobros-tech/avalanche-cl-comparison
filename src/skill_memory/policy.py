"""
Candidate selection and REUSE / CLONE / SCRATCH action policy.

Implements Sections 3 (steps 3-6), 3's "Core decision rule", and Section 8's
distinction between *compatibility scoring* ("how promising is this stored
skill?") and *action evaluation* ("will REUSE/CLONE actually beat SCRATCH
under a matched budget?").

Nothing in this module is allowed to see final evaluation/test labels
(Invariant 8) -- every estimate below is computed from a probe split that
the caller carves out of the *current experience's training data only*.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .registry import SkillInstance

REUSE, CLONE, SCRATCH = "REUSE", "CLONE", "SCRATCH"


@dataclass
class ActionEstimate:
    kind: str                       # REUSE | CLONE | SCRATCH
    expected_metric: float          # higher is better (probe accuracy)
    source: Optional[SkillInstance] # None for SCRATCH
    probe_trained_state: Optional[dict] = None  # reused to avoid re-training on selection


@torch.no_grad()
def _probe_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for x, y, *_ in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def _probe_train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    loss_fn: Callable = None,
) -> nn.Module:
    """Train `model` in place for a small, fixed ("matched") probe budget."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    return model


def select_candidates(
    scored_skills: List[tuple],  # (SkillInstance, score)
    score_threshold: float,
    top_k: int,
) -> List[SkillInstance]:
    """Section 3, step 4: filter + rank stored skills by compatibility score.

    This is explicitly a heuristic filter (Section 8: "A score threshold may
    be used only as an experimental candidate-filtering heuristic"), not the
    final decision. If nothing clears the bar, the caller ends up
    evaluating SCRATCH alone, which is the defined fallback.
    """
    filtered = [(s, sc) for s, sc in scored_skills if sc >= score_threshold]
    filtered.sort(key=lambda pair: pair[1], reverse=True)
    return [s for s, _ in filtered[:top_k]]


def estimate_reuse(
    candidate: SkillInstance,
    model_factory: Callable[[], nn.Module],
    probe_eval_loader: DataLoader,
    device: str,
) -> ActionEstimate:
    """REUSE never trains the candidate (Invariant 3/4)."""
    model = model_factory()
    model.load_state_dict(candidate.state_dict_copy())
    metric = _probe_accuracy(model, probe_eval_loader, device)
    return ActionEstimate(kind=REUSE, expected_metric=metric, source=candidate)


def estimate_clone(
    candidate: SkillInstance,
    model_factory: Callable[[], nn.Module],
    probe_train_loader: DataLoader,
    probe_eval_loader: DataLoader,
    probe_epochs: int,
    probe_lr: float,
    device: str,
) -> ActionEstimate:
    """CLONE: independent copy of the candidate + independent optimizer
    state, trained for the same matched probe budget as SCRATCH."""
    model = model_factory()
    model.load_state_dict(candidate.state_dict_copy())  # deep-copied state
    model = _probe_train(model, probe_train_loader, probe_epochs, probe_lr, device)
    metric = _probe_accuracy(model, probe_eval_loader, device)
    return ActionEstimate(
        kind=CLONE, expected_metric=metric, source=candidate,
        probe_trained_state=copy.deepcopy(model.state_dict()),
    )


def estimate_scratch(
    model_factory: Callable[[], nn.Module],
    probe_train_loader: DataLoader,
    probe_eval_loader: DataLoader,
    probe_epochs: int,
    probe_lr: float,
    device: str,
) -> ActionEstimate:
    model = model_factory()  # fresh init, independent optimizer state
    model = _probe_train(model, probe_train_loader, probe_epochs, probe_lr, device)
    metric = _probe_accuracy(model, probe_eval_loader, device)
    return ActionEstimate(
        kind=SCRATCH, expected_metric=metric, source=None,
        probe_trained_state=copy.deepcopy(model.state_dict()),
    )


def choose_best_action(
    actions: List[ActionEstimate],
    margin: float = 0.0,
) -> ActionEstimate:
    """Section 3's core decision rule:

        Choose REUSE or CLONE only when the evidence indicates it beats
        SCRATCH under the predefined objective. Otherwise choose SCRATCH.

    `margin` is an optional epsilon so that noisy probe estimates don't
    flip the decision on negligible differences -- it does not change the
    semantics, only the statistical conservatism of the comparison.
    """
    scratch = next(a for a in actions if a.kind == SCRATCH)
    non_scratch = [a for a in actions if a.kind != SCRATCH]

    best_alt = max(non_scratch, key=lambda a: a.expected_metric, default=None)

    if best_alt is not None and best_alt.expected_metric > scratch.expected_metric + margin:
        return best_alt
    return scratch
