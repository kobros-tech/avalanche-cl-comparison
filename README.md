# Avalanche CL Strategy Comparison + Skill Memory

This repository is the research-facing comparison harness for the Avalanche Skill Memory integration. It is no longer intended to be a toy/demo benchmark.

## Scientific protocol

The primary experiment follows the protocol used in the OCL Survey comparison work:

- Split CIFAR-100
- 20 class-incremental experiences
- replay memory size 2000 for reference replay methods
- five independent seeds: 0, 1, 2, 3, 4
- Python 3.11
- Avalanche 0.6.0
- CPU/GPU selected automatically by the workflow
- results retained as per-seed artifacts with provenance

The established reference strategies are executed from a pinned OCL Survey revision rather than reimplemented as toy versions in this repository: ER, ER-ACE, DER++, MIR, ER+LwF, RAR, SCR, A-GEM, MER, iCaRL, and GDumb. Skill Memory is evaluated using the Avalanche `SupervisedPlugin` implementation in `src/skill_memory/avalanche_plugin.py`.

The comparison is deliberately multi-seed. A strategy is not included in the primary aggregate table until all five requested seeds are available.

## Reproducible workflow

Run `.github/workflows/scientific-comparison.yml` from GitHub Actions. It creates a matrix over all reference strategies and seeds, runs Skill Memory through `experiments/run_skill_memory_scientific.py`, records provenance, and uploads a consolidated result artifact.

The workflow pins OCL Survey to commit `a0ecf4eb537bbe704598f1f423a52d0ca9d48a0f`, the revision containing the corrected result-processing/notebook work used as the reference for this comparison.

## Research notebooks

| Notebook | Purpose |
|---|---|
| `05_skill_memory_scientific.ipynb` | Five-seed Skill Memory analysis: final accuracy, causal forgetting, and acquisition decisions |
| `06_scientific_comparison.ipynb` | Aggregate every strategy from its own workflow artifacts and plot mean/std across seeds |

The older demo and synthetic notebooks are retained only as development/history material; they are not the source of the scientific comparison numbers.

## Evaluation discipline

The primary Skill Memory metric is **active-model** retention. The plugin's evaluation hooks may perform labeled probe retrieval for diagnostic experiments, but that oracle-routing behavior is intentionally excluded from the primary apples-to-apples table.

Forgetting is computed causally: for each task, the best accuracy observed before the final measurement is compared with the final accuracy. Future observations are never used to define the past maximum.

All raw per-seed outputs remain available as workflow artifacts, including Skill Memory's **REUSE/SCRATCH decisions** and probe diagnostics. This makes it possible to audit a surprising result rather than relying only on an aggregate number.

## Skill Memory implementation

`src/skill_memory/avalanche_plugin.py` contains the exact Avalanche integration under evaluation. It provides immutable stored skill snapshots, classifier resizing on restoration, probe-based compatibility measurements, dynamic gap-based skill selection with a forgetting guard, REUSE/SCRATCH acquisition, optimizer reset on restoration, and bounded skill registration.

There are **no fixed reuse/clone thresholds** in the scientific implementation. Skill selection is determined by the stored implementation's probe-based imagination procedure: the forgetting guard, largest-gap clustering, automatically derived floor, and joint score/accuracy candidate selection.

The implementation should be treated as experimental research code: the scientific workflow is designed to expose its behavior against established continual-learning baselines, not to assume that Skill Memory wins.

## Reference

The scientific comparison and result-processing protocol are based on the work in `kobros-tech/ocl_survey`, including its Split CIFAR-100 five-seed comparison notebook and corrected causal forgetting calculation.
