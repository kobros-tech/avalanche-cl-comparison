# Avalanche CL Strategy Comparison + Skill Memory

Compares the classic Continual Learning (CL) strategy families implemented
in [Avalanche](https://github.com/ContinualAI/avalanche) — regularization,
replay, and architectural/parameter-isolation methods — against a new
demo strategy, **Skill Memory**, proposed in
[`skill_memory_algorithm.md`](https://github.com/kobros-tech/avalanche/blob/feature/skill-memory-prototype/docs/skill_memory_algorithm.md)
(kobros-tech's fork of Avalanche).

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook notebooks/00_setup_and_benchmark.ipynb
```

Run the notebooks in order (`00` → `06`); each of `01`–`05` writes a
`results/*.csv` file that `06_comparison_dashboard.ipynb` aggregates.
They can also be run independently/out of order — each rebuilds its own
benchmark from the same seed.

## Notebooks

| # | Notebook | Strategies |
|---|----------|-----------|
| 00 | `00_setup_and_benchmark` | Environment check, benchmark exploration (PCA plot) |
| 01 | `01_baselines` | Naive, Cumulative, JointTraining |
| 02 | `02_regularization_strategies` | EWC, Synaptic Intelligence, LwF |
| 03 | `03_replay_strategies` | Replay, GEM, A-GEM, GDumb, **ER-ACE** |
| 04 | `04_architectural_strategies` | CWR\*, (AR1 noted but skipped — see below) |
| 05 | `05_skill_memory_demo` | **Skill Memory** (new): 3-level validation |
| 06 | `06_comparison_dashboard` | Aggregated comparison table + plots |

## Why a synthetic benchmark?

This project runs fully offline. `src/bench_utils.py` builds a small,
fast, in-memory class-incremental benchmark (Gaussian blobs per class,
split into disjoint-class experiences via Avalanche's own `nc_benchmark`)
instead of downloading MNIST/CIFAR/ImageNet. Every notebook only depends
on the resulting object having `.train_stream` / `.test_stream` /
`.n_classes` / `.feature_dim`, so swapping in a real dataset (e.g.
`avalanche.benchmarks.SplitMNIST(n_experiences=5)`) is a one-line change
— see `make_split_mnist_benchmark()` in `bench_utils.py`.

`src/skill_memory/synthetic_benchmark.py` is a second, independent
benchmark module used only by notebook 00 (basic exploration) and — via
its `make_transfer_demo_benchmark()` — internally by the Skill Memory
Level 2 experiment logic in notebook 05, which needs *repeated/perturbed*
classes across experiences (something Avalanche's `nc_benchmark` disjoint
class-incremental split can't produce) to demonstrate REUSE/CLONE
decisions with a known ground truth.

## What's *not* included, and why

- **Real datasets.** No network access to torchvision's dataset mirrors
  in this environment. All results here are on the synthetic benchmark;
  treat relative strategy rankings as illustrative, not as reproductions
  of published numbers.
- **AR1** is built around a convolutional feature extractor and doesn't
  have a sane default for flat feature-vector data, so notebook 04 notes
  the mismatch rather than forcing it onto data it wasn't designed for.
- **iCaRL, DER, MIR, RAR, SCR** — strong additional replay/rehearsal
  variants — are implemented in
  [`AlbinSou/ocl_survey`](https://github.com/AlbinSou/ocl_survey) (the
  code release for *"A Comprehensive Empirical Evaluation on Online
  Continual Learning,"* ICCVW 2023) but weren't ported here given the
  scope of this project. That repo is also a good source of literature
  hyperparameters (`config/best_configs/`) if you scale this project up
  to real datasets.

## Skill Memory (the new strategy)

Implemented in `src/skill_memory/`:

- `registry.py` — a `SkillMemory` that stores independent, **provably
  immutable** skill instances (deep-copied state dicts; a skill's stored
  weights can never change as a side effect of training a later skill).
- `scoring.py` — compatibility scoring: how well does a stored skill
  already do on a probe split of the *new* experience's training data
  (never test/eval data), relative to a fresh reference model?
- `policy.py` — the REUSE / CLONE / SCRATCH decision: estimate each
  action's outcome under a matched probe budget, and pick SCRATCH unless
  REUSE or CLONE genuinely beats it.
- `strategy.py` — `SkillMemoryStrategy`, a standalone orchestrator that
  runs the full algorithm against an Avalanche benchmark's experience
  streams (steps 1–10 of the spec's Section 3).
- `plugin.py` — a **not-yet-functional** sketch of how this would become
  a real `avalanche.core.SupervisedPlugin`, documenting the specific
  integration gap (no clean way to skip `optimizer.step()` for REUSE in
  Avalanche's current plugin hooks) worth raising upstream before
  finishing that path.

`tests/test_skill_memory.py` contains the Level-1 "mechanism validity"
unit tests (immutability, capacity limits, clone/source divergence,
REUSE never registering a new skill, etc.) — run with:

```bash
python -m pytest tests/ -v
```

`notebooks/05_skill_memory_demo.ipynb` runs three levels of validation
end to end, matching the spec's own suggested testing structure:

1. **Mechanism validity** — the same invariants as the unit tests, shown live.
2. **Oracle transfer check** — two experiences built to share the same
   underlying task; under a matched (tight) probe budget, the policy
   should (and does) pick REUSE over SCRATCH.
3. **Automatic policy on the shared benchmark** — run against the same
   disjoint-class benchmark every other notebook uses. Since classes
   never repeat there, the policy correctly falls back to SCRATCH every
   time — that's the spec-compliant behavior, not a bug.

### Known limitation, called out explicitly

Skill Memory as specified only defines the **training-time acquisition**
policy — not how to route a test example to the right stored skill at
**inference time** without an oracle task ID. Notebook 05's final
comparison uses a simple `class → skill_id` lookup table built from what
was learned during training, which only works because this specific demo
benchmark has disjoint classes. Real class-incremental inference (no
task ID at test time) would need Skill Memory's own inference-time
policy, which the current spec doesn't cover — a good next question to
raise with the spec's authors before comparing accuracy numbers as truly
apples-to-apples with the other strategies in the dashboard.

## Repo layout

```
requirements.txt
notebooks/
  00_setup_and_benchmark.ipynb
  01_baselines.ipynb
  02_regularization_strategies.ipynb
  03_replay_strategies.ipynb
  04_architectural_strategies.ipynb
  05_skill_memory_demo.ipynb
  06_comparison_dashboard.ipynb
src/
  bench_utils.py          # canonical shared benchmark (notebooks 01-06)
  run_utils.py             # shared train/eval loop + result-row helper
  cl_bench/
    er_ace.py               # ER-ACE, ported from AlbinSou/ocl_survey
  skill_memory/
    registry.py, scoring.py, policy.py, strategy.py, plugin.py
    synthetic_benchmark.py  # used by notebook 00 + Skill Memory's Level 2 demo
tests/
  test_skill_memory.py     # Level-1 mechanism validity tests
results/                   # *.csv + *.png written by the notebooks
```

## Contributing this upstream

If you want to take Skill Memory further toward an actual Avalanche PR:

1. Read `src/skill_memory/plugin.py`'s docstring — it documents the
   specific `SupervisedPlugin` hook points needed and the one concrete
   gap in Avalanche's current plugin API (skipping the optimizer step for
   REUSE) that's worth opening an issue about first.
2. The inference-time routing problem noted above is the other open
   design question worth raising with the `kobros-tech` branch's authors
   before finalizing a benchmark comparison.
