# Code from: https://github.com/Ladbaby/PyOmniTS
# Phase 2 metric tests implemented by GPT 5.6 SOL Extra High Model.
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.run_metrics import RunMetricsTracker, evaluation_tracker, read_model_nfe


class TinyContinuousModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.nfe = 0

    def forward(self, x):
        self.nfe = 4
        return self.linear(x)


class TestRunMetrics(unittest.TestCase):
    @staticmethod
    def _configs():
        return SimpleNamespace(
            save_run_metrics=1,
            dataset_name="TinyIrregular",
            dataset_id="TinyIrregular",
            model_name="TinyODE",
            model_id="TinyODE",
            task_name="short_term_forecast",
            seq_len=8,
            pred_len=2,
            missing_rate=0.25,
            missing_mechanism="auto",
            experiment_tag="phase2-test",
            itr_i=1,
            subfolder_train="test_run",
        )

    def test_training_and_evaluation_record(self):
        configs = self._configs()
        model = TinyContinuousModel()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "iter1"
            eval_dir = run_dir / "eval_1"
            tracker = RunMetricsTracker(configs, run_dir / "run_metrics.json")
            tracker.begin_training(model, "cpu")
            started = tracker.begin_train_epoch()
            model(torch.ones(3, 2)).sum().backward()
            tracker.observe_train_forward(model)
            tracker.end_train_epoch(started, iterations=1, epoch=0, train_stage=1)
            tracker.finish_training(model)

            evaluation = evaluation_tracker(configs, run_dir, eval_dir)
            evaluation.begin_evaluation(model, "cpu")
            started = evaluation.begin_inference_batch()
            model(torch.ones(3, 2))
            evaluation.end_inference_batch(started, model, batch_size=3)
            evaluation.finish_evaluation({"MSE": 0.125, "MAE": 0.25})

            with open(eval_dir / "run_metrics.json", "r", encoding="utf-8") as file:
                record = json.load(file)

        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["record_type"], "evaluation")
        self.assertEqual(record["run"]["missing_mechanism"], "random")
        self.assertEqual(record["run"]["seed"], 2025)
        self.assertEqual(record["model"]["parameters_total"], 6)
        self.assertEqual(record["training"]["completed_iterations"], 1)
        self.assertEqual(record["training"]["nfe_per_iteration"]["mean"], 4.0)
        self.assertGreater(record["training"]["iteration_time_ms"]["mean"], 0)
        self.assertEqual(record["evaluation"]["metrics"]["MSE"], 0.125)
        self.assertEqual(record["evaluation"]["nfe_per_batch"]["mean"], 4.0)
        self.assertIsNone(record["evaluation"]["peak_gpu_memory"])

    def test_nfe_is_optional(self):
        self.assertIsNone(read_model_nfe(torch.nn.Linear(2, 2)))
        self.assertEqual(read_model_nfe(TinyContinuousModel()), None)


if __name__ == "__main__":
    unittest.main()
