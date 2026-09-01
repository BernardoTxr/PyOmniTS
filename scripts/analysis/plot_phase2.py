# Code from: https://github.com/Ladbaby/PyOmniTS
# Phase 2 analysis implemented by GPT 5.6 SOL Extra High Model.
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .phase2_results import aggregate_rows, load_run_rows, write_rows_csv
except ImportError:
    from phase2_results import aggregate_rows, load_run_rows, write_rows_csv


CONDITION_FIELDS = (
    "dataset_name",
    "seq_len",
    "pred_len",
    "missing_mechanism",
    "missing_rate",
)
EFFICIENCY_FIELDS = CONDITION_FIELDS + ("batch_size", "hostname", "gpu_name")
GROUP_FIELDS = EFFICIENCY_FIELDS + ("model_name", "model_id")
COLORS = (
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#6C71D9",
)


def _publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _display_model(model_name: str, model_id: str | None = None) -> str:
    names = {
        "DTAMI_C": "DTAMI-C",
        "DTAMI_CIRC": "DTAMI-CIRC",
        "GRU_ODE_Bayes": "GRU-ODE-Bayes",
        "Neural_CDE": "Neural CDE",
        "Latent_ODE": "Latent ODE",
        "GNeuralFlow": "GNeuralFlow",
        "GraFITi": "GraFITi",
        "mTAN": "mTAN",
    }
    display = names.get(model_name, model_name.replace("_", " "))
    if model_id and model_id != model_name:
        display += f" ({model_id})"
    return display


def _model_label(row: dict[str, Any]) -> str:
    return _display_model(str(row["model_name"]), row.get("model_id"))


def _palette(rows: list[dict[str, Any]]) -> dict[str, str]:
    models = sorted({_model_label(row) for row in rows})
    return {model: COLORS[index % len(COLORS)] for index, model in enumerate(models)}


def _condition_title(condition: tuple[Any, ...]) -> str:
    dataset, seq_len, pred_len, mechanism, missing_rate = condition
    return f"{dataset}\nL={seq_len}, H={pred_len} | {mechanism}, q={float(missing_rate or 0):g}"


def _efficiency_title(condition: tuple[Any, ...]) -> str:
    title = _condition_title(condition[:len(CONDITION_FIELDS)])
    batch_size, hostname, gpu_name = condition[len(CONDITION_FIELDS):]
    hardware = gpu_name or hostname or "unknown hardware"
    return f"{title}\nbatch={batch_size} | {hardware}"


def _panels(
    rows: list[dict[str, Any]],
    required_fields: tuple[str, ...],
    max_panels: int,
    panel_fields: tuple[str, ...] = CONDITION_FIELDS,
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if all(row.get(field) is not None for field in required_fields):
            grouped[tuple(row.get(field) for field in panel_fields)].append(row)
    comparable = [
        item for item in grouped.items()
        if len({(row["model_name"], row.get("model_id")) for row in item[1]}) >= 2
    ]
    comparable.sort(key=lambda item: tuple(str(value) for value in item[0]))
    return comparable[:max_panels]


def _axes(count: int):
    columns = 1 if count == 1 else (2 if count <= 4 else 3)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.05 * rows), squeeze=False)
    flat = list(axes.flat)
    for axis in flat[count:]:
        axis.set_visible(False)
    return figure, flat[:count]


def _save(figure, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    paths = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(figure)
    return paths


def _aggregate_panel(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return aggregate_rows(rows, ("model_name", "model_id"))


def plot_quality(
    rows: list[dict[str, Any]], metric: str, output_dir: Path,
    formats: list[str], max_panels: int,
) -> list[str]:
    panels = _panels(rows, (metric,), max_panels)
    if not panels:
        return []
    colors = _palette(rows)
    figure, axes = _axes(len(panels))
    for axis, (condition, panel_rows) in zip(axes, panels):
        aggregated = _aggregate_panel(panel_rows, metric)
        aggregated.sort(key=lambda row: row.get(f"{metric}_mean", float("inf")))
        positions = np.arange(len(aggregated))
        means = np.asarray([row[f"{metric}_mean"] for row in aggregated])
        errors = np.asarray([
            1.96 * row.get(f"{metric}_std", 0.0) / math.sqrt(max(1, row.get(f"{metric}_n", 1)))
            for row in aggregated
        ])
        for position, mean, error, row in zip(positions, means, errors, aggregated):
            model = _model_label(row)
            axis.errorbar(position, mean, yerr=error, fmt="o", markersize=7,
                          color=colors[model], capsize=3, linewidth=1.2)
        axis.set_xticks(positions, [_model_label(row) for row in aggregated], rotation=30, ha="right")
        axis.set_ylabel(metric)
        axis.set_title(_condition_title(condition))
    figure.suptitle(f"Model comparison ({metric}; mean and 95% CI)", y=1.01, fontsize=12, fontweight="bold")
    figure.tight_layout()
    return _save(figure, output_dir, f"quality_{_safe_name(metric.lower())}", formats)


def _bubble_sizes(values: list[float | None], maximum: float | None) -> np.ndarray:
    if maximum is None or maximum <= 0:
        return np.full(len(values), 120.0)
    return np.asarray([
        55.0 if value is None else max(35.0, 520.0 * value / maximum)
        for value in values
    ])


def plot_tradeoff(
    rows: list[dict[str, Any]], metric: str, x_field: str, x_label: str,
    stem: str, output_dir: Path, formats: list[str], max_panels: int,
) -> list[str]:
    panels = _panels(rows, (metric, x_field), max_panels, EFFICIENCY_FIELDS)
    if not panels:
        return []
    colors = _palette(rows)
    all_memories = np.asarray([
        row["gpu_memory_gib"] for _, panel_rows in panels for row in panel_rows
        if row.get("gpu_memory_gib") is not None and row["gpu_memory_gib"] > 0
    ], dtype=float)
    maximum_memory = float(np.max(all_memories)) if all_memories.size else None
    figure, axes = _axes(len(panels))
    for axis, (condition, panel_rows) in zip(axes, panels):
        aggregated = aggregate_rows(panel_rows, ("model_name", "model_id"))
        aggregated = [row for row in aggregated if f"{x_field}_mean" in row and f"{metric}_mean" in row]
        x_values = [row[f"{x_field}_mean"] for row in aggregated]
        y_values = [row[f"{metric}_mean"] for row in aggregated]
        memories = [row.get("gpu_memory_gib_mean") for row in aggregated]
        sizes = _bubble_sizes(memories, maximum_memory)
        offsets = ((5, 7), (5, -22), (-5, 7), (-5, -22), (8, 18), (-8, 18))
        for index, row in enumerate(aggregated):
            model = _model_label(row)
            axis.scatter(x_values[index], y_values[index], s=sizes[index],
                         color=colors[model], alpha=0.82, edgecolor="white", linewidth=0.8)
            detail = model
            if memories[index] is not None:
                detail += f"\n{memories[index]:.2f} GiB"
            x_offset, y_offset = offsets[index % len(offsets)]
            axis.annotate(detail, (x_values[index], y_values[index]), xytext=(x_offset, y_offset),
                          textcoords="offset points", color=colors[model], fontsize=7,
                          fontweight="bold", ha="left" if x_offset > 0 else "right",
                          va="bottom" if y_offset > 0 else "top", clip_on=False)
        positive_x = [value for value in x_values if value > 0]
        if positive_x and max(positive_x) / min(positive_x) >= 20:
            axis.set_xscale("log")
        axis.margins(x=0.12, y=0.32)
        axis.set_xlabel(x_label)
        axis.set_ylabel(metric)
        axis.set_title(_efficiency_title(condition), fontsize=8)
    if all_memories.size:
        reference_memories = np.unique([np.min(all_memories), np.max(all_memories)])
        handles = [
            axes[0].scatter([], [], s=_bubble_sizes([float(memory)], maximum_memory)[0],
                            color="#9A9A9A", alpha=0.65, edgecolor="white", linewidth=0.8)
            for memory in reference_memories
        ]
        axes[0].legend(handles, [f"{memory:.2f} GiB" for memory in reference_memories],
                       title="Peak GPU memory", frameon=True, framealpha=0.92,
                       edgecolor="#CCCCCC", loc="best")
    figure.suptitle(f"Accuracy–efficiency trade-off (bubble area: peak GPU memory)",
                    y=1.01, fontsize=12, fontweight="bold")
    figure.tight_layout()
    return _save(figure, output_dir, stem, formats)


def _scenario_panels(
    rows: list[dict[str, Any]],
    varying: str,
    fixed_fields: tuple[str, ...],
    metric: str,
    max_panels: int,
):
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None and row.get(varying) is not None:
            grouped[tuple(row.get(field) for field in fixed_fields)].append(row)
    panels = []
    for key, group in grouped.items():
        model_keys = {(row.get("model_name"), row.get("model_id")) for row in group}
        if len({row.get(varying) for row in group}) >= 2 and len(model_keys) >= 2:
            panels.append((key, group))
    panels.sort(key=lambda item: tuple(str(value) for value in item[0]))
    return panels[:max_panels]


def plot_sensitivity(
    rows: list[dict[str, Any]], metric: str, varying: str,
    fixed_fields: tuple[str, ...], x_label: str, stem: str,
    title: Callable[[tuple[Any, ...]], str], output_dir: Path,
    formats: list[str], max_panels: int,
) -> list[str]:
    panels = _scenario_panels(rows, varying, fixed_fields, metric, max_panels)
    if not panels:
        return []
    colors = _palette(rows)
    figure, axes = _axes(len(panels))
    for axis, (key, panel_rows) in zip(axes, panels):
        aggregated = aggregate_rows(panel_rows, ("model_name", "model_id", varying))
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in aggregated:
            by_model[_model_label(row)].append(row)
        for model, model_rows in sorted(by_model.items()):
            model_rows.sort(key=lambda row: float(row[varying]))
            x = np.asarray([float(row[varying]) for row in model_rows])
            y = np.asarray([row[f"{metric}_mean"] for row in model_rows])
            error = np.asarray([
                1.96 * row.get(f"{metric}_std", 0.0) / math.sqrt(max(1, row.get(f"{metric}_n", 1)))
                for row in model_rows
            ])
            axis.plot(x, y, marker="o", color=colors[model], linewidth=1.7, label=model)
            axis.fill_between(x, y - error, y + error, color=colors[model], alpha=0.13, linewidth=0)
        axis.set_xlabel(x_label)
        axis.set_ylabel(metric)
        axis.set_title(title(key))
        axis.legend(frameon=False)
    figure.tight_layout()
    return _save(figure, output_dir, stem, formats)


def _filter_rows(rows: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if args.datasets and row.get("dataset_name") not in args.datasets:
            continue
        if args.models and row.get("model_name") not in args.models:
            continue
        if args.seq_len is not None and row.get("seq_len") is None:
            continue
        if args.seq_len is not None and int(row["seq_len"]) != args.seq_len:
            continue
        if args.pred_len is not None and row.get("pred_len") is None:
            continue
        if args.pred_len is not None and int(row["pred_len"]) != args.pred_len:
            continue
        if args.missing_mechanisms and row.get("missing_mechanism") not in args.missing_mechanisms:
            continue
        if args.batch_size is not None and row.get("batch_size") != args.batch_size:
            continue
        if args.gpu_name is not None and row.get("gpu_name") != args.gpu_name:
            continue
        if args.hostname is not None and row.get("hostname") != args.hostname:
            continue
        if args.experiment_tag is not None and row.get("experiment_tag", "") != args.experiment_tag:
            continue
        output.append(row)
    return output


def build_charts(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _publication_style()
    generated: dict[str, list[str]] = {}
    generated["quality"] = plot_quality(rows, args.metric, output_dir, args.formats, args.max_panels)
    generated["training_tradeoff"] = plot_tradeoff(
        rows, args.metric, "train_iteration_ms", "Training time (ms / iteration)",
        "tradeoff_training_time", output_dir, args.formats, args.max_panels,
    )
    generated["nfe_tradeoff"] = plot_tradeoff(
        rows, args.metric, "time_per_nfe_ms", "Time (ms / NFE)",
        "tradeoff_time_per_nfe", output_dir, args.formats, args.max_panels,
    )
    generated["parameter_tradeoff"] = plot_tradeoff(
        rows, args.metric, "parameters_trainable", "Trainable parameters",
        "tradeoff_parameters", output_dir, args.formats, args.max_panels,
    )
    generated["missingness"] = plot_sensitivity(
        rows, args.metric, "missing_rate",
        ("dataset_name", "seq_len", "pred_len", "missing_mechanism"),
        "Injected missing rate", "sensitivity_missingness",
        lambda key: f"{key[0]} ({key[3]}; context={key[1]}, horizon={key[2]})",
        output_dir, args.formats, args.max_panels,
    )
    generated["horizon"] = plot_sensitivity(
        rows, args.metric, "pred_len",
        ("dataset_name", "seq_len", "missing_mechanism", "missing_rate"),
        "Forecast horizon", "sensitivity_horizon",
        lambda key: f"{key[0]} ({key[2]}, missing={float(key[3] or 0):g}; context={key[1]})",
        output_dir, args.formats, args.max_panels,
    )
    manifest = {
        "metric": args.metric,
        "rows": len(rows),
        "generated": generated,
        "skipped": [name for name, paths in generated.items() if not paths],
        "note": "A chart is skipped when fewer than two comparable models or fewer than two scenario levels are available.",
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create publication-style Phase 2 comparison charts.")
    parser.add_argument("--results-root", default="storage/results")
    parser.add_argument("--metrics-root", default="metrics", help="Legacy timing/memory JSON root.")
    parser.add_argument("--output-dir", default="storage/analysis/phase2")
    parser.add_argument("--metric", default="MSE")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--pred-len", type=int)
    parser.add_argument("--missing-mechanisms", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpu-name")
    parser.add_argument("--hostname")
    parser.add_argument("--experiment-tag")
    parser.add_argument("--max-panels", type=int, default=12)
    parser.add_argument("--all-evaluations", action="store_true", help="Treat repeated evaluations of one checkpoint as separate rows.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = load_run_rows(args.results_root, args.metrics_root, args.all_evaluations)
    rows = _filter_rows(rows, args)
    if not rows:
        raise SystemExit("No matching metric records were found.")
    output_dir = Path(args.output_dir)
    write_rows_csv(rows, output_dir / "phase2_runs.csv")
    aggregate = aggregate_rows(rows, GROUP_FIELDS)
    write_rows_csv(aggregate, output_dir / "phase2_summary.csv")
    manifest = build_charts(rows, args)
    generated_count = sum(len(paths) for paths in manifest["generated"].values())
    print(f"Loaded {len(rows)} runs; wrote {generated_count} chart files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
