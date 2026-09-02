# Code from: https://github.com/Ladbaby/PyOmniTS
# Phase 2 publication tables implemented by GPT 5.6 SOL Extra High Model.
import argparse
import csv
import math
from pathlib import Path
from typing import Any

try:
    from .phase2_results import aggregate_rows, load_run_rows
except ImportError:
    from phase2_results import aggregate_rows, load_run_rows


MODEL_LABELS = {
    "DTAMI_C": "DTAMI-C",
    "DTAMI_CIRC": "DTAMI-CIRC",
    "GRU_D": "GRU-D",
}

OUTPUT_COLUMNS = (
    ("model", "Model"),
    ("runs", "Runs"),
    ("mse", "MSE"),
    ("mae", "MAE"),
    ("train_ms", "Training (ms/iter)"),
    ("inference_ms", "Inference (ms/sample)"),
    ("memory_gib", "Peak GPU memory (GiB)"),
    ("parameters", "Trainable parameters"),
)


def _filter(rows: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if args.dataset and row.get("dataset_name") != args.dataset:
            continue
        if args.models and row.get("model_name") not in args.models:
            continue
        if args.seq_len is not None and int(row.get("seq_len", -1)) != args.seq_len:
            continue
        if args.pred_len is not None and int(row.get("pred_len", -1)) != args.pred_len:
            continue
        if args.experiment_tag and row.get("experiment_tag") != args.experiment_tag:
            continue
        output.append(row)
    return output


def _mean_std(row: dict[str, Any], field: str, digits: int) -> str:
    mean = row.get(f"{field}_mean")
    if mean is None:
        return "--"
    std = row.get(f"{field}_std", 0.0)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _parameter_count(row: dict[str, Any]) -> str:
    value = row.get("parameters_trainable_mean")
    if value is None:
        return "--"
    return f"{int(round(value)):,}"


def build_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated = aggregate_rows(rows, ("model_name",))
    aggregated.sort(key=lambda row: row.get("MSE_mean", math.inf))
    table = []
    for rank, row in enumerate(aggregated, start=1):
        table.append({
            "rank": rank,
            "model": MODEL_LABELS.get(row["model_name"], row["model_name"]),
            "runs": int(row["runs"]),
            "mse": _mean_std(row, "MSE", 4),
            "mae": _mean_std(row, "MAE", 4),
            "train_ms": _mean_std(row, "train_iteration_ms", 2),
            "inference_ms": _mean_std(row, "inference_sample_ms", 3),
            "memory_gib": _mean_std(row, "gpu_memory_gib", 2),
            "parameters": _parameter_count(row),
        })
    return table


def _markdown(table: list[dict[str, Any]]) -> str:
    headings = ["Rank"] + [label for _, label in OUTPUT_COLUMNS]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join(["---:", "---"] + ["---:"] * (len(headings) - 2)) + " |",
    ]
    for row in table:
        values = [str(row["rank"])] + [str(row[key]) for key, _ in OUTPUT_COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", r"\_").replace("±", r"$\pm$")


def _latex(table: list[dict[str, Any]]) -> str:
    headings = ["Rank"] + [label for _, label in OUTPUT_COLUMNS]
    alignment = "r" + "l" + "r" * (len(headings) - 2)
    lines = [
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(_latex_escape(value) for value in headings) + r" \\",
        r"\midrule",
    ]
    for row in table:
        values = [row["rank"]] + [row[key] for key, _ in OUTPUT_COLUMNS]
        lines.append(" & ".join(_latex_escape(value) for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create Phase 2 publication tables.")
    parser.add_argument("--results-root", default="storage/results")
    parser.add_argument("--metrics-root", default="metrics")
    parser.add_argument("--output-dir", default="storage/analysis/phase2")
    parser.add_argument("--dataset")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--pred-len", type=int)
    parser.add_argument("--experiment-tag")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = _filter(load_run_rows(args.results_root, args.metrics_root), args)
    if not rows:
        raise SystemExit("No matching metric records were found.")
    table = build_table(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "performance_table.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("rank",) + tuple(key for key, _ in OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(table)
    with open(output_dir / "performance_table.md", "w", encoding="utf-8") as file:
        file.write(_markdown(table))
    with open(output_dir / "performance_table.tex", "w", encoding="utf-8") as file:
        file.write(_latex(table))

    print(_markdown(table), end="")
    print(f"Wrote publication tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
