# Code from: https://github.com/Ladbaby/PyOmniTS
# Phase 2 analysis implemented by GPT 5.6 SOL Extra High Model.
import csv
import json
import math
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


IDENTITY_FIELDS = (
    "dataset_name",
    "dataset_id",
    "model_name",
    "model_id",
    "task_name",
    "seq_len",
    "pred_len",
    "missing_rate",
    "missing_mechanism",
    "experiment_tag",
    "iteration",
    "seed",
)


def _get(payload: dict[str, Any], *keys: str, default=None):
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _record_row(record: dict[str, Any], path: Path) -> dict[str, Any] | None:
    metrics = _get(record, "evaluation", "metrics", default={})
    if not metrics:
        return None
    run = record.get("run", {})
    config = record.get("config", {})
    row = {field: run.get(field) for field in IDENTITY_FIELDS}
    row.update({
        "record_path": str(path),
        "run_directory": str(path.parent.parent if path.parent.name.startswith("eval_") else path.parent),
        "parameters_total": _number(_get(record, "model", "parameters_total")),
        "parameters_trainable": _number(_get(record, "model", "parameters_trainable")),
        "model_size_mib": _number(_get(record, "model", "parameter_and_buffer_size_mib")),
        "train_iteration_ms": _number(_get(record, "training", "iteration_time_ms", "mean")),
        "train_nfe": _number(_get(record, "training", "nfe_per_iteration", "mean")),
        "train_time_per_nfe_ms": _number(_get(record, "training", "wall_time_per_nfe_ms")),
        "inference_batch_ms": _number(_get(record, "evaluation", "steady_state_batch_time_ms", "mean")),
        "inference_sample_ms": _number(_get(record, "evaluation", "steady_state_sample_time_ms")),
        "inference_nfe": _number(_get(record, "evaluation", "nfe_per_batch", "mean")),
        "inference_time_per_nfe_ms": _number(_get(record, "evaluation", "steady_state_time_per_nfe_ms")),
        "gpu_memory_gib": _number(
            _get(record, "training", "peak_gpu_memory", "allocated_gib")
            or _get(record, "evaluation", "peak_gpu_memory", "allocated_gib")
        ),
        "hostname": _get(record, "environment", "hostname"),
        "gpu_name": _get(record, "environment", "gpu_name"),
        "batch_size": config.get("batch_size"),
        "allow_tf32": config.get("allow_tf32"),
        "legacy_record": False,
    })
    row["time_per_nfe_ms"] = (
        row["inference_time_per_nfe_ms"] or row["train_time_per_nfe_ms"]
    )
    for name, value in metrics.items():
        number = _number(value)
        if number is not None:
            row[name] = number
    return row


def _find_config(metric_path: Path) -> Path | None:
    for parent in metric_path.parents:
        candidate = parent / "configs.yaml"
        if candidate.exists():
            return candidate
        if parent.name.startswith("iter"):
            break
    return None


def _legacy_row(metric_path: Path) -> dict[str, Any] | None:
    config_path = _find_config(metric_path)
    if config_path is None:
        return None
    try:
        metrics = _read_json(metric_path)
        with open(config_path, "r", encoding="utf-8") as file:
            configs = yaml.safe_load(file) or {}
    except (OSError, json.JSONDecodeError, yaml.YAMLError):
        return None
    iteration_name = config_path.parent.name
    iteration = configs.get("itr_i", 0)
    if iteration_name.startswith("iter") and iteration_name[4:].isdigit():
        iteration = int(iteration_name[4:])
    missing_rate = configs.get("missing_rate", 0.0)
    mechanism = configs.get("missing_mechanism", "auto")
    if mechanism == "auto":
        mechanism = "random" if float(missing_rate) > 0 else "native"
    row = {
        "dataset_name": configs.get("dataset_name"),
        "dataset_id": configs.get("dataset_id"),
        "model_name": configs.get("model_name"),
        "model_id": configs.get("model_id"),
        "task_name": configs.get("task_name"),
        "seq_len": configs.get("seq_len"),
        "pred_len": configs.get("pred_len"),
        "missing_rate": missing_rate,
        "missing_mechanism": mechanism,
        "experiment_tag": configs.get("experiment_tag", ""),
        "enc_in": configs.get("enc_in"),
        "batch_size": configs.get("batch_size"),
        "allow_tf32": configs.get("allow_tf32"),
        "iteration": iteration,
        "seed": 2024 + int(iteration),
        "record_path": str(metric_path),
        "run_directory": str(config_path.parent),
        "legacy_record": True,
    }
    for name, value in metrics.items():
        number = _number(value)
        if number is not None:
            row[name] = number
    return row


def _legacy_efficiency(rows: list[dict[str, Any]], metrics_root: Path) -> None:
    task_files = list(metrics_root.glob("*/model_*.json")) if metrics_root.exists() else []
    stores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in task_files:
        try:
            stores[path.name].append(_read_json(path))
        except (OSError, json.JSONDecodeError):
            continue

    for row in rows:
        dataset = str(row.get("dataset_name"))
        model_id = str(row.get("model_id"))
        seq_len = row.get("seq_len")
        pred_len = row.get("pred_len")
        input_key = f"{seq_len}/{pred_len}"
        complexity_key = f"seq_len_{seq_len}_enc_in_"

        for store in stores.get("model_gpu_memories.json", []):
            value = _get(store, dataset, input_key, model_id)
            if _number(value) is not None:
                if row.get("gpu_memory_gib") is None:
                    row["gpu_memory_gib"] = _number(value)

        train_stores = stores.get("model_train_time.json", [])
        for store in train_stores:
            host_order = [socket.gethostname()] + [host for host in store if host != socket.gethostname()]
            for host in host_order:
                value = _get(store, host, dataset, input_key, model_id)
                if _number(value) is not None:
                    if row.get("train_iteration_ms") is None:
                        row["train_iteration_ms"] = _number(value)
                    row.setdefault("hostname", host)
                    break

        for store in stores.get("model_complexities.json", []):
            enc_in = row.get("enc_in")
            exact_key = f"seq_len_{seq_len}_enc_in_{enc_in}" if enc_in is not None else None
            for key, models in store.items():
                if (key == exact_key or (exact_key is None and key.startswith(complexity_key))) and model_id in models:
                    if row.get("parameters_total") is None:
                        row["parameters_total"] = _number(models[model_id].get("params"))
                    break


def load_run_rows(
    results_root: Path | str,
    metrics_root: Path | str | None = None,
    keep_all_evaluations: bool = False,
) -> list[dict[str, Any]]:
    """Load new records and fall back to legacy metric.json/configs.yaml pairs."""
    results_root = Path(results_root)
    rows: list[dict[str, Any]] = []
    recorded_evaluation_directories: set[Path] = set()
    for path in sorted(results_root.rglob("run_metrics.json")):
        try:
            row = _record_row(_read_json(path), path)
        except (OSError, json.JSONDecodeError):
            continue
        if row is not None:
            rows.append(row)
            recorded_evaluation_directories.add(path.parent.resolve())

    for path in sorted(results_root.rglob("metric.json")):
        if path.parent.resolve() in recorded_evaluation_directories:
            continue
        row = _legacy_row(path)
        if row is not None:
            rows.append(row)

    if metrics_root is not None:
        _legacy_efficiency(rows, Path(metrics_root))

    if not keep_all_evaluations:
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(Path(row["run_directory"]).resolve())
            previous = latest.get(key)
            if previous is None or row["record_path"] > previous["record_path"]:
                latest[key] = row
        rows = list(latest.values())
    return sorted(rows, key=lambda row: (
        str(row.get("dataset_name")),
        str(row.get("model_name")),
        int(row.get("iteration") or 0),
    ))


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)

    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result = dict(zip(group_fields, key))
        numeric_fields = sorted({
            name for row in group for name, value in row.items()
            if _number(value) is not None and name not in group_fields
        })
        for name in numeric_fields:
            values = np.asarray([
                _number(row.get(name)) for row in group if _number(row.get(name)) is not None
            ], dtype=float)
            if values.size:
                result[f"{name}_mean"] = float(np.mean(values))
                result[f"{name}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                result[f"{name}_n"] = int(values.size)
        result["runs"] = len(group)
        output.append(result)
    return output


def write_rows_csv(rows: Iterable[dict[str, Any]], path: Path | str) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(IDENTITY_FIELDS)
    fields.extend(sorted({key for row in rows for key in row if key not in fields}))
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
