"""
Shared "run this strategy over this benchmark and give me a tidy result
row" helper, so every notebook logs results in the same format into
results/*.csv and 06_comparison_dashboard.ipynb can just concatenate them.
"""
from __future__ import annotations

import io
import time
from typing import Optional

import torch


def run_strategy(
    strategy,
    benchmark,
    strategy_name: str,
    category: str,
    eval_every_exp: bool = True,
):
    """Train `strategy` sequentially over `benchmark.train_stream`,
    evaluating on the full `test_stream` after every experience, and
    return (per_experience_rows, final_summary_row).

    `strategy` must already be constructed (model/optimizer/plugins etc
    attached) -- this function only drives the standard Avalanche
    train/eval loop and harvests metrics, it doesn't configure strategies
    itself (that stays explicit in each notebook, since the whole point is
    to compare how each strategy is set up).
    """
    per_exp_rows = []
    t0 = time.time()

    for i, exp in enumerate(benchmark.train_stream):
        strategy.train(exp)
        if eval_every_exp:
            res = strategy.eval(benchmark.test_stream)
            stream_acc = _get_metric(res, "Top1_Acc_Stream")
            stream_forgetting = _get_metric(res, "StreamForgetting")
            per_exp_rows.append({
                "strategy": strategy_name,
                "category": category,
                "after_experience": i,
                "stream_acc": stream_acc,
                "stream_forgetting": stream_forgetting,
            })

    if not eval_every_exp:
        res = strategy.eval(benchmark.test_stream)
        per_exp_rows.append({
            "strategy": strategy_name,
            "category": category,
            "after_experience": len(benchmark.train_stream) - 1,
            "stream_acc": _get_metric(res, "Top1_Acc_Stream"),
            "stream_forgetting": _get_metric(res, "StreamForgetting"),
        })

    elapsed = time.time() - t0
    final = dict(per_exp_rows[-1])
    final["train_seconds"] = elapsed
    return per_exp_rows, final


def _get_metric(results: dict, prefix: str) -> Optional[float]:
    for k, v in results.items():
        if k.startswith(prefix):
            return float(v)
    return None
