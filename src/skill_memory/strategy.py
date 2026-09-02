"""
SkillMemoryStrategy: orchestrates the sequential algorithm in Section 3 of
skill_memory_algorithm.md against an Avalanche benchmark's train_stream /
test_stream.

This is a *standalone orchestrator* (Level 1-3 experimental harness from
Section 12), not yet a `SupervisedTemplate` subclass -- it manages its own
models and training loops directly against Avalanche `experience.dataset`
objects so every invariant in Section 7 stays easy to audit line-by-line.

`plugin.py` in this package sketches how this would be wrapped as a real
`SupervisedPlugin` for an eventual Avalanche PR (Section 13's integration
boundary): Avalanche keeps owning the stream/optimizer/eval lifecycle,
Skill Memory keeps owning the registry + policy.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from .policy import (
    CLONE,
    REUSE,
    SCRATCH,
    choose_best_action,
    estimate_clone,
    estimate_reuse,
    estimate_scratch,
    select_candidates,
)
from .registry import SkillInstance, SkillMemory
from .scoring import compatibility_score


def state_dict_hash(state_dict: dict) -> str:
    """Cheap fingerprint used to *prove* immutability in tests/notebooks:
    if a stored skill's hash ever changes, something violated Invariant 1."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


@dataclass
class DecisionRecord:
    experience_id: int
    decision: str                       # REUSE | CLONE | SCRATCH
    chosen_skill_id: Optional[int]      # skill that now "owns" this experience
    source_skill_id: Optional[int]      # non-None only for CLONE
    scores: dict                        # {skill_id: compatibility_score}
    probe_metrics: dict                 # {"REUSE:<id>": acc, "CLONE:<id>": acc, "SCRATCH": acc}
    final_test_acc: float
    n_skills_after: int


class SkillMemoryStrategy:
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        max_skills: int,
        device: str = "cpu",
        score_threshold: float = 0.3,
        top_k_candidates: int = 2,
        probe_subset_size: int = 200,
        probe_epochs: int = 1,
        probe_lr: float = 0.05,
        full_epochs: int = 3,
        full_lr: float = 0.05,
        batch_size: int = 64,
        decision_margin: float = 0.01,
        seed: int = 0,
    ):
        self.model_factory = model_factory
        self.memory = SkillMemory(max_skills=max_skills)
        self.device = device
        self.score_threshold = score_threshold
        self.top_k_candidates = top_k_candidates
        self.probe_subset_size = probe_subset_size
        self.probe_epochs = probe_epochs
        self.probe_lr = probe_lr
        self.full_epochs = full_epochs
        self.full_lr = full_lr
        self.batch_size = batch_size
        self.decision_margin = decision_margin
        self.seed = seed

        self.decisions: List[DecisionRecord] = []
        # experience_id -> skill_id that "owns" (handles) that experience
        self.experience_to_skill: dict = {}

    # ------------------------------------------------------------------ #
    def _make_probe_loaders(self, train_dataset):
        """Carve REUSE/CLONE/SCRATCH decision-making probe splits *only*
        out of the current experience's training data (Invariant 8: no
        test/eval labels here, ever)."""
        n = len(train_dataset)
        probe_n = min(self.probe_subset_size, n // 2 if n > 1 else n)
        probe_n = max(probe_n, 2)
        remainder = n - probe_n
        gen = torch.Generator().manual_seed(self.seed)
        probe_ds, _ = random_split(train_dataset, [probe_n, remainder], generator=gen)
        gen2 = torch.Generator().manual_seed(self.seed + 1)
        half = max(1, probe_n // 2)
        probe_train_ds, probe_eval_ds = random_split(
            probe_ds, [half, probe_n - half], generator=gen2
        )
        probe_train_loader = DataLoader(probe_train_ds, batch_size=self.batch_size, shuffle=True)
        probe_eval_loader = DataLoader(probe_eval_ds, batch_size=self.batch_size, shuffle=False)
        return probe_train_loader, probe_eval_loader

    def _full_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _train_full(self, model: nn.Module, loader: DataLoader) -> nn.Module:
        model.to(self.device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.full_lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.full_epochs):
            for x, y, *_ in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
        return model

    @torch.no_grad()
    def _eval(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        model.to(self.device)
        correct, total = 0, 0
        for x, y, *_ in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    # ------------------------------------------------------------------ #
    def process_experience(self, experience_id: int, train_dataset, test_dataset,
                            classes_seen: List[int]) -> DecisionRecord:
        """Section 3, steps 2-10, run for a single experience.

        We reseed torch's *global* RNG deterministically per experience
        because several DataLoaders below use `shuffle=True` without an
        explicit generator, which would otherwise draw on whatever global
        RNG state happens to be left over from prior training calls --
        breaking the "matched comparison" fairness the spec requires
        between REUSE / CLONE / SCRATCH probe estimates (Invariant 12).
        """
        torch.manual_seed(self.seed * 100_003 + experience_id)
        probe_train_loader, probe_eval_loader = self._make_probe_loaders(train_dataset)
        full_train_loader = self._full_loader(train_dataset)
        full_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        scores = {}
        candidates = []
        if not self.memory.is_empty:
            reference_model = self.model_factory()  # fresh init, used only as scoring anchor
            for skill in self.memory.all_skills():
                m = self.model_factory()
                m.load_state_dict(skill.state_dict_copy())
                s = compatibility_score(m, reference_model, probe_eval_loader, device=self.device)
                scores[skill.skill_id] = s
            candidates = select_candidates(
                [(self.memory.get(sid), sc) for sid, sc in scores.items()],
                score_threshold=self.score_threshold,
                top_k=self.top_k_candidates,
            )

        probe_metrics = {}
        if not candidates:
            # Step 4: nothing promising enough -> SCRATCH directly, no need
            # to run probe estimates at all.
            decision_kind, decision_source = SCRATCH, None
        else:
            actions = []
            for cand in candidates:
                r = estimate_reuse(cand, self.model_factory, probe_eval_loader, self.device)
                c = estimate_clone(
                    cand, self.model_factory, probe_train_loader, probe_eval_loader,
                    self.probe_epochs, self.probe_lr, self.device,
                )
                probe_metrics[f"REUSE:{cand.skill_id}"] = r.expected_metric
                probe_metrics[f"CLONE:{cand.skill_id}"] = c.expected_metric
                actions.extend([r, c])
            sc = estimate_scratch(
                self.model_factory, probe_train_loader, probe_eval_loader,
                self.probe_epochs, self.probe_lr, self.device,
            )
            probe_metrics["SCRATCH"] = sc.expected_metric
            actions.append(sc)

            best = choose_best_action(actions, margin=self.decision_margin)
            decision_kind = best.kind
            decision_source = best.source

        # --- Execute the decision (Section 3, steps 7-9) -----------------
        if decision_kind == REUSE:
            chosen_skill = decision_source
            final_model = self.model_factory()
            final_model.load_state_dict(chosen_skill.state_dict_copy())  # unchanged, no training
            source_skill_id = None

        elif decision_kind == CLONE:
            final_model = self.model_factory()
            final_model.load_state_dict(decision_source.state_dict_copy())
            final_model = self._train_full(final_model, full_train_loader)
            chosen_skill = self.memory.register(
                name=f"skill_{experience_id}_clone_of_{decision_source.skill_id}",
                model_state=final_model.state_dict(),
                acquisition_mode="clone",
                experience_id=experience_id,
                source_id=decision_source.skill_id,
                compatibility_score=scores.get(decision_source.skill_id),
                classes_seen=classes_seen,
            )
            source_skill_id = decision_source.skill_id

        else:  # SCRATCH
            final_model = self.model_factory()
            final_model = self._train_full(final_model, full_train_loader)
            chosen_skill = self.memory.register(
                name=f"skill_{experience_id}_scratch",
                model_state=final_model.state_dict(),
                acquisition_mode="scratch",
                experience_id=experience_id,
                classes_seen=classes_seen,
            )
            source_skill_id = None

        final_test_acc = self._eval(final_model, full_test_loader)
        self.experience_to_skill[experience_id] = chosen_skill.skill_id

        record = DecisionRecord(
            experience_id=experience_id,
            decision=decision_kind,
            chosen_skill_id=chosen_skill.skill_id,
            source_skill_id=source_skill_id,
            scores=scores,
            probe_metrics=probe_metrics,
            final_test_acc=final_test_acc,
            n_skills_after=len(self.memory),
        )
        self.decisions.append(record)
        return record

    # ------------------------------------------------------------------ #
    def evaluate_all_skills(self, test_datasets_by_experience: dict) -> dict:
        """Section 11.4 preservation check: re-evaluate every stored skill
        on the test set of the experience that originally produced it.
        Since stored states are provably immutable, any accuracy drift here
        would indicate a bug, not "forgetting" -- true forgetting in this
        design would instead show up as `estimate_clone` needing many
        epochs, or as the policy repeatedly choosing SCRATCH."""
        results = {}
        for skill in self.memory.all_skills():
            test_ds = test_datasets_by_experience.get(skill.experience_id)
            if test_ds is None:
                continue
            m = self.model_factory()
            m.load_state_dict(skill.state_dict_copy())
            loader = DataLoader(test_ds, batch_size=256, shuffle=False)
            results[skill.skill_id] = self._eval(m, loader)
        return results

    def assert_no_mutation(self, hashes_before: dict) -> List[int]:
        """Return the list of skill_ids whose stored state changed since
        `hashes_before` was captured (should always be empty)."""
        violated = []
        for skill in self.memory.all_skills():
            if skill.skill_id not in hashes_before:
                continue
            if state_dict_hash(skill.model_state) != hashes_before[skill.skill_id]:
                violated.append(skill.skill_id)
        return violated

    def capture_hashes(self) -> dict:
        return {s.skill_id: state_dict_hash(s.model_state) for s in self.memory.all_skills()}
