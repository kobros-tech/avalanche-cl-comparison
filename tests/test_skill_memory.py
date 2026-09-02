"""
Level 1 - Mechanism validity (Section 12 of skill_memory_algorithm.md):

    REUSE   -> source unchanged
    CLONE   -> source unchanged + clone changes
    SCRATCH -> new independent skill state

Run with:  python -m pytest tests/test_skill_memory.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from skill_memory import SkillMemory, state_dict_hash
from skill_memory.policy import estimate_reuse, estimate_clone, estimate_scratch
from torch.utils.data import TensorDataset, DataLoader


def tiny_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def tiny_loader(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=4)


# --------------------------------------------------------------------- #
def test_registration_deep_copies_state():
    mem = SkillMemory(max_skills=4)
    model = tiny_model()
    skill = mem.register("s0", model.state_dict(), acquisition_mode="scratch", experience_id=0)

    # mutate the ORIGINAL model after registering...
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    # ...the stored skill must be untouched (Invariant 1 / 2).
    for k, v in skill.model_state.items():
        assert not torch.allclose(v, model.state_dict()[k])


def test_capacity_enforced():
    mem = SkillMemory(max_skills=1)
    model = tiny_model()
    mem.register("s0", model.state_dict(), acquisition_mode="scratch", experience_id=0)
    with pytest.raises(RuntimeError):
        mem.register("s1", model.state_dict(), acquisition_mode="scratch", experience_id=1)


def test_reuse_never_mutates_source_and_creates_no_new_skill():
    mem = SkillMemory(max_skills=4)
    model = tiny_model()
    skill = mem.register("s0", model.state_dict(), acquisition_mode="scratch", experience_id=0)
    h_before = state_dict_hash(skill.model_state)
    n_before = len(mem)

    loader = tiny_loader()
    estimate_reuse(skill, tiny_model, loader, device="cpu")

    assert state_dict_hash(skill.model_state) == h_before, "REUSE must not mutate the source skill"
    assert len(mem) == n_before, "REUSE must not create a new skill record"


def test_clone_leaves_source_unchanged_and_clone_differs():
    mem = SkillMemory(max_skills=4)
    model = tiny_model()
    skill = mem.register("s0", model.state_dict(), acquisition_mode="scratch", experience_id=0)
    h_before = state_dict_hash(skill.model_state)

    train_loader = tiny_loader(seed=1)
    eval_loader = tiny_loader(seed=2)
    est = estimate_clone(
        skill, tiny_model, train_loader, eval_loader,
        probe_epochs=3, probe_lr=0.5, device="cpu",
    )

    # Source must be provably untouched.
    assert state_dict_hash(skill.model_state) == h_before

    # The clone's *trained* weights should differ from the source (after
    # several SGD steps at a large LR, equality would indicate the clone
    # never actually trained).
    differs = any(
        not torch.allclose(est.probe_trained_state[k], skill.model_state[k])
        for k in skill.model_state
    )
    assert differs, "Cloned+trained model should diverge from its frozen source"


def test_scratch_is_independent_of_any_existing_skill():
    mem = SkillMemory(max_skills=4)
    model = tiny_model()
    skill = mem.register("s0", model.state_dict(), acquisition_mode="scratch", experience_id=0)

    train_loader = tiny_loader(seed=3)
    eval_loader = tiny_loader(seed=4)
    est = estimate_scratch(tiny_model, train_loader, eval_loader,
                            probe_epochs=1, probe_lr=0.1, device="cpu")

    # A freshly initialized + probe-trained model should not accidentally
    # equal the unrelated stored skill's parameters.
    same = all(
        torch.allclose(est.probe_trained_state[k], skill.model_state[k])
        for k in skill.model_state
    )
    assert not same


def test_registration_after_clone_records_lineage():
    mem = SkillMemory(max_skills=4)
    source = mem.register("s0", tiny_model().state_dict(), acquisition_mode="scratch", experience_id=0)
    clone = mem.register(
        "s1", tiny_model().state_dict(), acquisition_mode="clone",
        experience_id=1, source_id=source.skill_id, compatibility_score=0.9,
    )
    assert clone.source_id == source.skill_id
    assert clone.acquisition_mode == "clone"


def test_reuse_acquisition_mode_rejected_by_registry():
    """REUSE must never call register() at all (Section 2 / Invariant 5)."""
    mem = SkillMemory(max_skills=4)
    with pytest.raises(ValueError):
        mem.register("bad", tiny_model().state_dict(), acquisition_mode="reuse", experience_id=0)


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
