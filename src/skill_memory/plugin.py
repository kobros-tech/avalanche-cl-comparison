"""
Sketch of a real Avalanche `SupervisedPlugin` integration (Section 13:
"Avalanche integration boundary").

This is deliberately NOT wired up to a training loop yet -- it documents
the hook points a PR to ContinualAI/avalanche would need, so the mapping
from the standalone `SkillMemoryStrategy` (strategy.py) to a first-class
Avalanche `SupervisedTemplate` plugin is explicit:

  before_training_exp:
      - run compatibility scoring + candidate selection + action estimation
        using `strategy.model.state_dict()`'s architecture as the
        `model_factory`.
      - if the decision is REUSE: load the chosen skill's state into
        `strategy.model` and set a flag so `after_training_iteration`
        skips optimizer.step() for this experience (REUSE must not train).
      - if CLONE: load the source skill's state into `strategy.model`
        before Avalanche's normal training loop runs, and give
        `strategy.optimizer` a fresh state (Invariant 10: clone optimizer
        isolation) -- e.g. by re-instantiating it.
      - if SCRATCH: reset `strategy.model` to a fresh init before training.

  after_training_exp:
      - if the decision was CLONE or SCRATCH, deep-copy
        `strategy.model.state_dict()` into a new `SkillInstance` and
        register it (Invariant 5: only CLONE/SCRATCH create skills).
      - if REUSE, log the event but do not touch the registry.

  after_eval_exp:
      - optionally re-evaluate previously registered skills for the
        preservation report (Section 11.4).

Real integration would also need a strategy-level "skip training" switch
for REUSE (Avalanche doesn't currently have a public toggle for "run the
eval-mode forward pass but never call optimizer.step()"), which is exactly
the kind of gap worth raising as an issue/PR upstream before finishing
this plugin.
"""
from __future__ import annotations

from typing import Callable, Optional

from torch import nn

from .registry import SkillMemory

try:
    from avalanche.core import SupervisedPlugin
except ImportError:  # pragma: no cover - avalanche is an optional dependency here
    SupervisedPlugin = object


class SkillMemoryPlugin(SupervisedPlugin):
    """NOT YET FUNCTIONAL -- see module docstring. Included as a starting
    point for anyone picking this up to contribute upstream."""

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        max_skills: int,
        score_threshold: float = 0.3,
        top_k_candidates: int = 2,
    ):
        super().__init__()
        self.model_factory = model_factory
        self.memory = SkillMemory(max_skills=max_skills)
        self.score_threshold = score_threshold
        self.top_k_candidates = top_k_candidates
        self._pending_decision: Optional[str] = None

    def before_training_exp(self, strategy, **kwargs):
        raise NotImplementedError(
            "See module docstring: wire this up to strategy.model / "
            "strategy.optimizer once you've settled on how this repo "
            "wants to expose a 'skip training for this experience' hook."
        )

    def after_training_exp(self, strategy, **kwargs):
        raise NotImplementedError
