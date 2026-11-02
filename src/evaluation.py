"""Standalone evaluation / reporting script.

Reads the model comparison and final metrics reports produced by the
training pipeline and prints a formatted summary.

Usage
-----
    python -m src.evaluation

Report sources
--------------
* ``reports/model_comparison.csv``   \u2013 produced by ``python -m src.train``
* ``reports/model_metrics.json``     \u2013 produced by ``python -m src.train``
* ``reports/classification_comparison.csv``  \u2013 produced by ``python -m src.classification``
  (optional; used only if the canonical comparison is missing)
"""

from pathlib import Path
import json
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Columns emitted by src.train (canonical) and src.classification (extra).
_METRIC_COLUMNS = [
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_pr_auc",
]


def _load_comparison() -> pd.DataFrame:
    """Return the model comparison DataFrame.

    Prefers the canonical ``model_comparison.csv`` produced by ``src.train``.
    Falls back to ``classification_comparison.csv`` produced by
    ``src.classification`` so the script still works when only the optional
    comparison has been run.
    """
    candidates = [
        REPORTS / "model_comparison.csv",
        REPORTS / "classification_comparison.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    print(f"Reading comparison from: {path.name}")
                    return df
            except Exception as exc:
                print(
                    f"WARNING: could not read {path.name}: {exc}",
                    file=sys.stderr,
                )
    return pd.DataFrame()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model comparison table
    # ------------------------------------------------------------------
    comparison = _load_comparison()

    if comparison.empty:
        print(
            "No model comparison report found.\n"
            "Run:  python -m src.train\n"
            "  or: python -m src.classification",
            file=sys.stderr,
        )
    else:
        # Determine sort column robustly.
        sort_col = next(
            (
                col
                for col in [
                    "validation_pr_auc",
                    "pr_auc",
                    "validation_f1",
                    "f1",
                ]
                if col in comparison.columns
            ),
            None,
        )
        if sort_col:
            comparison = comparison.sort_values(sort_col, ascending=False)

        print("\n===== CLASSIFICATION MODEL COMPARISON =====")
        print(comparison.to_string(index=False))

        # Plot only columns that are both numeric and present.
        plot_cols = [c for c in _METRIC_COLUMNS if c in comparison.columns]
        if not plot_cols:
            # Fallback to any numeric column that looks like a metric.
            plot_cols = [
                c
                for c in comparison.select_dtypes(include="number").columns
                if any(
                    kw in c.lower()
                    for kw in ("precision", "recall", "f1", "auc")
                )
            ]

        model_col = next(
            (
                c
                for c in ["model", "model_name", "algorithm"]
                if c in comparison.columns
            ),
            None,
        )

        if model_col and plot_cols:
            ax = (
                comparison.set_index(model_col)[plot_cols]
                .plot(kind="bar", figsize=(12, 6))
            )
            ax.set_title("Classification Model Comparison (validation metrics)")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            out_path = REPORTS / "model_comparison.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"\nSaved bar chart to {out_path}")

    # ------------------------------------------------------------------
    # Final test metrics from the training run
    # ------------------------------------------------------------------
    metrics_path = REPORTS / "model_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                report = json.load(f)

            selected = report.get("selected_model", "unknown")
            threshold = report.get("decision_threshold", "N/A")
            test_m = report.get("test_metrics", {})

            print(f"\n===== FINAL TEST METRICS (model: {selected}) =====")
            print(f"Decision threshold : {threshold}")
            for key, value in test_m.items():
                print(f"  {key:15s}: {value}")
        except Exception as exc:
            print(
                f"WARNING: could not read {metrics_path.name}: {exc}",
                file=sys.stderr,
            )
    else:
        print(
            f"\nNo {metrics_path.name} found. Run: python -m src.train",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
