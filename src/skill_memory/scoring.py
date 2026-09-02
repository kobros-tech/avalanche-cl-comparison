"""
Compatibility scoring (Section 8 of skill_memory_algorithm.md).

    score = clamp(1 - L_skill / L_ref, 0, 1)

`L_skill` is the loss of a *stored* skill (loaded read-only, no gradient
step) on a small probe split drawn from the *training* data of the new
experience. `L_ref` is a reference loss with no learned signal at all
(a freshly-initialized model of the same architecture), which anchors the
score at ~0 when the stored skill is no better than a random model.

Per Invariant 8, this function must never see final evaluation/test labels
-- callers are responsible for only passing it a probe split carved out of
the experience's *training* stream.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _average_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: str,
) -> float:
    model.eval()
    model.to(device)
    total_loss, total_n = 0.0, 0
    for x, y, *_ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / max(total_n, 1)


def compatibility_score(
    skill_model: nn.Module,
    reference_model: nn.Module,
    probe_loader: DataLoader,
    loss_fn: Callable = None,
    device: str = "cpu",
) -> float:
    """Compute score = clamp(1 - L_skill / L_ref, 0, 1).

    Both models are evaluated in eval() / no_grad mode on the *same* probe
    split, so no gradient updates and no test labels are ever touched here.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    l_skill = _average_loss(skill_model, probe_loader, loss_fn, device)
    l_ref = _average_loss(reference_model, probe_loader, loss_fn, device)

    if l_ref <= 1e-12:
        # Degenerate reference loss -- fall back to a neutral score rather
        # than dividing by (near) zero.
        return 0.0

    score = 1.0 - (l_skill / l_ref)
    return float(min(1.0, max(0.0, score)))
