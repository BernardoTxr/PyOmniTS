# Code from: https://github.com/Ladbaby/PyOmniTS
# Phase 2 analysis tests implemented by GPT 5.6 SOL Extra High Model.
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.phase2_results import load_run_rows
from scripts.analysis.plot_phase2 import main


def _write_record(
    results_root: Path,
    model: str,
    iteration: int,
    pred_len: int,
    missing_rate: float,
):
    run_dir = (
        results_root / "Tiny" / "Tiny" / model / model /
        f"12_{pred_len}" / f"missing_{missing_rate}" / f"iter{iteration}"
    )
    eval_dir = run_dir / "eval_1"
    eval_dir.mkdir(parents=True)
    model_offset = 0.0 if model == "DTAMI_C" else 0.025
    mse = 0.08 + model_offset + 0.12 * missing_rate + 0.001 * pred_len + iteration * 0.002
    record = {
        "schema_version": 1,
        "record_type": "evaluation",
        "config": {"batch_size": 4, "allow_tf32": 0},
        "run": {
            "dataset_name": "Tiny",
            "dataset_id": "Tiny",
            "model_name": model,
            "model_id": model,
            "task_name": "short_term_forecast",
            "seq_len": 12,
            "pred_len": pred_len,
            "missing_rate": missing_rate,
            "missing_mechanism": "random",
            "experiment_tag": "phase2-test",
            "iteration": iteration,
            "seed": 2024 + iteration,
        },
        "environment": {"hostname": "test", "gpu_name": "Test GPU"},
        "model": {
            "parameters_total": 1000 if model == "DTAMI_C" else 1400,
            "parameters_trainable": 1000 if model == "DTAMI_C" else 1400,
            "parameter_and_buffer_size_mib": 0.01,
        },
        "training": {
            "iteration_time_ms": {"mean": 10.0 if model == "DTAMI_C" else 14.0},
            "nfe_per_iteration": {"mean": 4.0},
            "wall_time_per_nfe_ms": 2.5 if model == "DTAMI_C" else 3.5,
            "peak_gpu_memory": {"allocated_gib": 0.5 if model == "DTAMI_C" else 0.8},
        },
        "evaluation": {
            "metrics": {"MSE": mse, "MAE": mse ** 0.5},
            "steady_state_batch_time_ms": {"mean": 4.0 if model == "DTAMI_C" else 6.0},
            "nfe_per_batch": {"mean": 4.0},
            "steady_state_time_per_nfe_ms": 1.0 if model == "DTAMI_C" else 1.5,
            "peak_gpu_memory": {"allocated_gib": 0.3},
        },
    }
    with open(eval_dir / "run_metrics.json", "w", encoding="utf-8") as file:
        json.dump(record, file)


class TestPhase2Analysis(unittest.TestCase):
    def test_collector_and_all_publication_charts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            output = root / "charts"
            for model in ("DTAMI_C", "Neural_CDE"):
                for iteration in (0, 1):
                    for pred_len in (2, 4):
                        for missing_rate in (0.1, 0.4):
                            _write_record(results, model, iteration, pred_len, missing_rate)

            rows = load_run_rows(results)
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(row["MSE"] > 0 for row in rows))
            self.assertTrue(all(row["time_per_nfe_ms"] > 0 for row in rows))
            self.assertTrue(all(row["batch_size"] == 4 for row in rows))

            result = main([
                "--results-root", str(results),
                "--metrics-root", str(root / "no-legacy-metrics"),
                "--output-dir", str(output),
                "--formats", "png",
                "--experiment-tag", "phase2-test",
            ])

            self.assertEqual(result, 0)
            expected = (
                "quality_mse.png",
                "tradeoff_training_time.png",
                "tradeoff_time_per_nfe.png",
                "tradeoff_parameters.png",
                "sensitivity_missingness.png",
                "sensitivity_horizon.png",
                "phase2_runs.csv",
                "phase2_summary.csv",
                "manifest.json",
            )
            for name in expected:
                self.assertTrue((output / name).exists(), name)
                self.assertGreater((output / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
