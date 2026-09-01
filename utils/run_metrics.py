# Code from: https://github.com/Ladbaby/PyOmniTS
# Phase 2 metric recording implemented by GPT 5.6 SOL Extra High Model.
import datetime
import json
import math
import os
import platform
import socket
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import Module

from utils.globals import accelerator, logger


GIB = 1024 ** 3


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Convert common scientific-Python values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _summary(values: list[float]) -> dict[str, float | int] | None:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if clean.size == 0:
        return None
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "median": float(np.median(clean)),
        "p95": float(np.percentile(clean, 95)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def _atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(_jsonable(payload), file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(temporary_path, path)


def _config_dict(configs: Any) -> dict[str, Any]:
    if is_dataclass(configs):
        return asdict(configs)
    return dict(vars(configs))


def _resolved_missing_mechanism(configs: Any) -> str:
    mechanism = getattr(configs, "missing_mechanism", "auto")
    if mechanism != "auto":
        return mechanism
    return "random" if float(getattr(configs, "missing_rate", 0.0)) > 0 else "native"


def _model_counts(model: Module) -> dict[str, int | float]:
    model = accelerator.unwrap_model(model)
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
    return {
        "parameters_total": int(total),
        "parameters_trainable": int(trainable),
        "parameter_and_buffer_size_mib": float((parameter_bytes + buffer_bytes) / (1024 ** 2)),
    }


def read_model_nfe(model: Module) -> float | None:
    """Read the latest forward-pass NFE from models that expose ``model.nfe``."""
    model = accelerator.unwrap_model(model)
    try:
        value = getattr(model, "nfe")
        if callable(value):
            value = value()
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


class RunMetricsTracker:
    """Collect reproducibility and efficiency metrics without changing model APIs."""

    schema_version = 1

    def __init__(
        self,
        configs: Any,
        record_path: Path | str,
        base_record_path: Path | str | None = None,
    ):
        self.configs = configs
        self.record_path = Path(record_path)
        self.enabled = bool(
            getattr(configs, "save_run_metrics", 1) and accelerator.is_main_process
        )
        self.device = torch.device("cpu")
        self._train_epochs: list[dict[str, Any]] = []
        self._train_nfe: list[float] = []
        self._eval_timings: list[tuple[float, int, float | None]] = []
        self._training_started_at: float | None = None

        self.record = self._new_record()
        if base_record_path is not None and Path(base_record_path).exists():
            try:
                with open(base_record_path, "r", encoding="utf-8") as file:
                    self.record = json.load(file)
                self.record["record_type"] = "evaluation"
                self.record["evaluation"] = {}
            except (OSError, json.JSONDecodeError) as error:
                logger.warning("Could not reuse run metrics from %s: %s", base_record_path, error)

    def _new_record(self) -> dict[str, Any]:
        configs = _config_dict(self.configs)
        iteration = int(getattr(self.configs, "itr_i", 0))
        return {
            "schema_version": self.schema_version,
            "record_type": "training",
            "created_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "run": {
                "dataset_name": getattr(self.configs, "dataset_name", None),
                "dataset_id": getattr(self.configs, "dataset_id", None),
                "model_name": getattr(self.configs, "model_name", None),
                "model_id": getattr(self.configs, "model_id", None),
                "task_name": getattr(self.configs, "task_name", None),
                "seq_len": getattr(self.configs, "seq_len", None),
                "pred_len": getattr(self.configs, "pred_len", None),
                "missing_rate": getattr(self.configs, "missing_rate", 0.0),
                "missing_mechanism": _resolved_missing_mechanism(self.configs),
                "experiment_tag": getattr(self.configs, "experiment_tag", ""),
                "iteration": iteration,
                "seed": 2024 + iteration,
                "training_folder": getattr(self.configs, "subfolder_train", ""),
            },
            "config": configs,
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            "model": {},
            "training": {},
            "evaluation": {},
        }

    def _sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _reset_peak_memory(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            self._sync()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_memory(self) -> dict[str, float] | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        self._sync()
        return {
            "allocated_gib": float(torch.cuda.max_memory_allocated(self.device) / GIB),
            "reserved_gib": float(torch.cuda.max_memory_reserved(self.device) / GIB),
        }

    def _record_model_and_environment(self, model: Module, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.record["model"].update(_model_counts(model))
        environment = self.record["environment"]
        environment["device"] = str(self.device)
        if self.device.type == "cuda" and torch.cuda.is_available():
            environment["gpu_name"] = torch.cuda.get_device_name(self.device)
            environment["gpu_total_memory_gib"] = float(
                torch.cuda.get_device_properties(self.device).total_memory / GIB
            )

    def save(self) -> None:
        if not self.enabled:
            return
        self.record["updated_at_utc"] = _utc_now()
        try:
            _atomic_json_dump(self.record_path, self.record)
        except OSError as error:
            logger.warning("Could not save run metrics to %s: %s", self.record_path, error)

    def begin_training(self, model: Module, device: torch.device | str) -> None:
        if not self.enabled:
            return
        self._record_model_and_environment(model, device)
        self._reset_peak_memory()
        self._sync()
        self._training_started_at = time.perf_counter()
        self.save()

    def begin_train_epoch(self) -> float | None:
        if not self.enabled:
            return None
        self._sync()
        return time.perf_counter()

    def observe_train_forward(self, model: Module) -> None:
        if not self.enabled:
            return
        nfe = read_model_nfe(model)
        if nfe is not None:
            self._train_nfe.append(nfe)

    def end_train_epoch(
        self,
        started_at: float | None,
        iterations: int,
        epoch: int,
        train_stage: int,
    ) -> None:
        if not self.enabled or started_at is None or iterations <= 0:
            return
        self._sync()
        duration_seconds = time.perf_counter() - started_at
        self._train_epochs.append({
            "epoch": int(epoch),
            "train_stage": int(train_stage),
            "iterations": int(iterations),
            "duration_seconds": float(duration_seconds),
            "mean_iteration_ms": float(duration_seconds * 1000 / iterations),
        })

    def finish_training(self, model: Module) -> None:
        if not self.enabled:
            return
        self._record_model_and_environment(model, self.device)
        self._sync()
        total_seconds = sum(epoch["duration_seconds"] for epoch in self._train_epochs)
        total_iterations = sum(epoch["iterations"] for epoch in self._train_epochs)
        epoch_iteration_times = [epoch["mean_iteration_ms"] for epoch in self._train_epochs]
        iteration_summary = _summary(epoch_iteration_times)
        if iteration_summary is not None:
            iteration_summary["count"] = int(total_iterations)
            iteration_summary["mean"] = float(total_seconds * 1000 / total_iterations)
            iteration_summary["epochs_measured"] = len(self._train_epochs)
            iteration_summary["measurement"] = (
                "wall-clock training epoch divided by optimizer iterations; includes data loading, "
                "forward, loss, backward, and optimizer step"
            )

        training = self.record["training"]
        training.update({
            "completed_epochs": len(self._train_epochs),
            "completed_iterations": int(total_iterations),
            "measured_wall_time_seconds": float(total_seconds),
            "iteration_time_ms": iteration_summary,
            "nfe_per_iteration": _summary(self._train_nfe),
            "peak_gpu_memory": self._peak_memory(),
        })
        if self._training_started_at is not None:
            training["total_wall_time_seconds"] = float(
                time.perf_counter() - self._training_started_at
            )
        if total_seconds > 0 and self._train_nfe:
            training["wall_time_per_nfe_ms"] = float(
                total_seconds * 1000 / sum(self._train_nfe)
            )
        self.save()

    def begin_evaluation(self, model: Module, device: torch.device | str) -> None:
        if not self.enabled:
            return
        self.record["record_type"] = "evaluation"
        self._record_model_and_environment(model, device)
        self._reset_peak_memory()

    def begin_inference_batch(self) -> float | None:
        if not self.enabled:
            return None
        self._sync()
        return time.perf_counter()

    def end_inference_batch(
        self,
        started_at: float | None,
        model: Module,
        batch_size: int,
    ) -> None:
        if not self.enabled or started_at is None:
            return
        self._sync()
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        self._eval_timings.append((elapsed_ms, int(batch_size), read_model_nfe(model)))

    def finish_evaluation(self, metrics: dict[str, Any] | None) -> None:
        if not self.enabled:
            return
        timings = self._eval_timings
        steady_timings = timings[1:] if len(timings) > 1 else timings
        batch_times = [item[0] for item in timings]
        steady_batch_times = [item[0] for item in steady_timings]
        total_samples = sum(item[1] for item in steady_timings)
        total_nfe = sum(item[2] for item in steady_timings if item[2] is not None)
        evaluation = self.record["evaluation"]
        evaluation.update({
            "metrics": _jsonable(metrics or {}),
            "timed_batches": len(timings),
            "warmup_batches_excluded": 1 if len(timings) > 1 else 0,
            "batch_time_ms": _summary(batch_times),
            "steady_state_batch_time_ms": _summary(steady_batch_times),
            "nfe_per_batch": _summary([
                item[2] for item in steady_timings if item[2] is not None
            ]),
            "peak_gpu_memory": self._peak_memory(),
        })
        if total_samples > 0:
            evaluation["steady_state_sample_time_ms"] = float(
                sum(steady_batch_times) / total_samples
            )
        if total_nfe > 0:
            evaluation["steady_state_time_per_nfe_ms"] = float(
                sum(steady_batch_times) / total_nfe
            )
        self.save()


def evaluation_tracker(
    configs: Any,
    run_directory: Path | str,
    evaluation_directory: Path | str,
) -> RunMetricsTracker:
    """Create an evaluation snapshot that reuses metrics from its training run."""
    run_directory = Path(run_directory)
    evaluation_directory = Path(evaluation_directory)
    tracker = RunMetricsTracker(
        configs=configs,
        record_path=evaluation_directory / "run_metrics.json",
        base_record_path=run_directory / "run_metrics.json",
    )
    tracker.record["run"]["evaluation_folder"] = evaluation_directory.name
    if run_directory.name.startswith("iter") and run_directory.name[4:].isdigit():
        iteration = int(run_directory.name[4:])
        tracker.record["run"]["iteration"] = iteration
        tracker.record["run"]["seed"] = 2024 + iteration
    return tracker
