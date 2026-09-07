#!/usr/bin/env python3
"""Run the canonical OCL Survey Skill Memory implementation on Split-CIFAR100.

The plugin implementation in src/skill_memory/avalanche_plugin.py is copied
verbatim from OCL Survey's feature/skill-memory-comparison-notebook branch.
This runner only provides the comparison experiment and evaluates the active
model directly on the test stream; it does not introduce label-based skill
retrieval during evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from avalanche.benchmarks import with_classes_timeline
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics


def load_plugin_class():
    path = Path(__file__).resolve().parents[1] / "src" / "skill_memory" / "avalanche_plugin.py"
    module_name = "comparison_skill_memory_plugin"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load canonical Skill Memory plugin from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.SkillMemoryPlugin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ocl-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def accuracy(model, dataset, device, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.numel())
    return correct / total if total else float("nan")


def main():
    args = parse_args()
    ocl_root = args.ocl_root.resolve()
    sys.path.insert(0, str(ocl_root))

    from src.factories import benchmark_factory, model_factory
    from src.factories.benchmark_factory import DS_SIZES

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    configured = args.device.lower()
    device = "cuda" if configured == "auto" and torch.cuda.is_available() else configured
    if configured == "auto" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    benchmark = benchmark_factory.create_benchmark(
        benchmark_name="split_cifar100",
        n_experiences=20,
        val_size=0.05,
        use_transforms=True,
        fixed_class_order=None,
        dataset_root=str(ocl_root / "data"),
    )

    model = model_factory.create_model(model_type="resnet18", input_size=DS_SIZES["split_cifar100"]).to(torch_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)

    SkillMemoryPlugin = load_plugin_class()
    plugin = SkillMemoryPlugin(
        max_skills=20,
        forgetting_margin=0.05,
        score_floor=None,
        probe_batch_size=64,
        probe_batches=5,
        probe_seed=args.seed,
    )

    evaluator = EvaluationPlugin(accuracy_metrics(stream=True), loggers=[])
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=10,
        train_epochs=3,
        eval_mb_size=128,
        evaluator=evaluator,
        plugins=[plugin],
        device=torch_device,
    )

    accuracy_history = []
    decisions = []
    for t, experience in enumerate(benchmark.train_stream):
        strategy.train(with_classes_timeline([experience]), reset_optimizer_state=False, num_workers=0, drop_last=True)
        row = {"after_experience": t}
        for j, test_exp in enumerate(benchmark.test_stream[: t + 1]):
            row[f"task_{j}"] = accuracy(strategy.model, test_exp.dataset, torch_device)
        row["stream_accuracy"] = float(np.nanmean([v for k, v in row.items() if k.startswith("task_")]))
        accuracy_history.append(row)
        decisions.append({
            "experience": t,
            "decision": plugin.last_decision,
            "selected_skill": plugin.last_selected_skill,
            "compatibility_score": plugin.last_compatibility_score,
            "old_accuracy": plugin.last_old_accuracy,
            "new_accuracy": plugin.last_new_accuracy,
        })
        print(f"seed={args.seed} exp={t:02d} decision={plugin.last_decision} stream_acc={row['stream_accuracy']:.4f}", flush=True)

    matrix = np.full((len(accuracy_history), len(benchmark.test_stream)), np.nan)
    for i, row in enumerate(accuracy_history):
        for j in range(len(benchmark.test_stream)):
            if f"task_{j}" in row:
                matrix[i, j] = row[f"task_{j}"]

    forgetting = []
    for task in range(matrix.shape[1]):
        observed = matrix[:, task]
        observed = observed[~np.isnan(observed)]
        if len(observed) > 1:
            forgetting.append(float(np.max(observed[:-1]) - observed[-1]))

    summary = {
        "method": "Skill Memory",
        "benchmark": "split_cifar100",
        "n_experiences": 20,
        "memory_size": 2000,
        "seed": args.seed,
        "final_accuracy": float(np.nanmean(matrix[-1])),
        "forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "AAA_test": float(np.mean([r["stream_accuracy"] for r in accuracy_history])),
        "decisions": decisions,
        "accuracy_history": accuracy_history,
        "evaluation_mode": "active_model_no_retrieval",
        "probe": {"batch_size": 64, "batches": 5, "seed": args.seed},
        "skill_memory_source": "ocl_survey/src/strategies/skill_memory.py",
        "skill_memory_source_sha": "35dc4b9355255d961deb621ebf9926a4aa55b8ac",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("seed", "final_accuracy", "forgetting", "AAA_test")}, indent=2))


if __name__ == "__main__":
    main()
