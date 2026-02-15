from __future__ import annotations

import pandas as pd

from gsrd.eval.statistics import mean_ci95, summarize_granularity


def test_mean_ci95_singleton() -> None:
    mean, std, lo, hi = mean_ci95([0.25])
    assert mean == 0.25
    assert std == 0.0
    assert lo == 0.25
    assert hi == 0.25


def test_summarize_granularity_outputs_robustness_columns() -> None:
    df = pd.DataFrame(
        [
            {"detector": "d", "dataset": "s", "granularity": "standard", "vocab_id": "a", "mAP": 0.6, "mAP50": 0.8},
            {"detector": "d", "dataset": "s", "granularity": "standard", "vocab_id": "b", "mAP": 0.5, "mAP50": 0.7},
            {"detector": "d", "dataset": "s", "granularity": "fine", "vocab_id": "c", "mAP": 0.3, "mAP50": 0.5},
        ]
    )
    out = summarize_granularity(df)
    assert "mAP_ci95_low" in out.columns
    assert "mAP_ci95_high" in out.columns
    assert "drop_from_standard_mAP" in out.columns
    assert "robustness_to_standard" in out.columns
    fine = out[out["granularity"] == "fine"].iloc[0]
    assert fine["drop_from_standard_mAP"] > 0
