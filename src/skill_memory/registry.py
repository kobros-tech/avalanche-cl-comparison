"""
Skill Memory registry.

Implements Section 2, 6, 7 and 10 of skill_memory_algorithm.md:

    - A skill instance is an independently stored model state + metadata.
    - `max_skills` counts stored skill instances, not neurons/parameters.
    - A registered source skill never changes as a side effect of another
      skill's training (Invariant 1).
    - Only CLONE and SCRATCH acquisitions create new skill records
      (Invariant 5 / Section 10).

This module is intentionally model-agnostic: it stores whatever
`state_dict()`-like object the caller gives it and hands back deep copies.
It does not know about neurons, layers, or parameter ownership -- see
Section 6 of the spec for why that is a deliberate design choice.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillInstance:
    """An independently stored, immutable learned skill."""

    skill_id: int
    name: str
    model_state: Dict[str, Any]          # deep-copied state_dict
    acquisition_mode: str                # "scratch" | "clone"
    source_id: Optional[int]             # set only when acquisition_mode == "clone"
    compatibility_score: Optional[float] # score that led to this skill being chosen as a CLONE source (None for scratch)
    experience_id: int                   # index of the experience that produced this skill
    creation_order: int                  # monotonically increasing counter
    classes_seen: List[int] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def state_dict_copy(self) -> Dict[str, Any]:
        """Return a fresh deep copy of this skill's parameters.

        Callers MUST use this (never the raw `model_state`) so that nothing
        downstream can accidentally mutate the registry's stored state.
        """
        return copy.deepcopy(self.model_state)


class SkillMemory:
    """Registry of independently stored skill instances.

    Enforces:
      - Immutability: `register()` always deep-copies the incoming state,
        so later in-place mutation of the caller's model can never leak
        into a stored skill.
      - Capacity accounting: registration fails once `max_skills` stored
        skills exist (Invariant 6).
      - Monotonic acquisition: skills are appended, never overwritten
        (Invariant 7).
    """

    def __init__(self, max_skills: int):
        if max_skills < 1:
            raise ValueError("max_skills must be >= 1")
        self.max_skills = max_skills
        self._skills: List[SkillInstance] = []
        self._id_counter = itertools.count()
        self._order_counter = itertools.count()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._skills)

    @property
    def is_full(self) -> bool:
        return len(self._skills) >= self.max_skills

    @property
    def is_empty(self) -> bool:
        return len(self._skills) == 0

    def all_skills(self) -> List[SkillInstance]:
        """Return the list of stored skills (the SkillInstance objects
        themselves, not copies -- callers must use `state_dict_copy()`
        rather than mutating `.model_state` directly)."""
        return list(self._skills)

    def get(self, skill_id: int) -> SkillInstance:
        for s in self._skills:
            if s.skill_id == skill_id:
                return s
        raise KeyError(f"No skill with id={skill_id}")

    # ------------------------------------------------------------------ #
    # Registration (Section 10)
    # ------------------------------------------------------------------ #
    def register(
        self,
        name: str,
        model_state: Dict[str, Any],
        acquisition_mode: str,
        experience_id: int,
        source_id: Optional[int] = None,
        compatibility_score: Optional[float] = None,
        classes_seen: Optional[List[int]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> SkillInstance:
        if acquisition_mode not in ("scratch", "clone"):
            raise ValueError("acquisition_mode must be 'scratch' or 'clone' "
                              "(REUSE must never call register())")
        if self.is_full:
            raise RuntimeError(
                f"SkillMemory is full (max_skills={self.max_skills}); "
                "cannot register a new skill."
            )
        if acquisition_mode == "clone" and source_id is None:
            raise ValueError("source_id is required when acquisition_mode='clone'")

        skill = SkillInstance(
            skill_id=next(self._id_counter),
            name=name,
            model_state=copy.deepcopy(model_state),  # <-- immutability guarantee
            acquisition_mode=acquisition_mode,
            source_id=source_id,
            compatibility_score=compatibility_score,
            experience_id=experience_id,
            creation_order=next(self._order_counter),
            classes_seen=list(classes_seen or []),
            extra=extra or {},
        )
        self._skills.append(skill)
        return skill
