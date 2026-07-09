import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_mamba_native.tiny_overfit import run_finetuning_comparison


class TestTinyFinetuningComparison(unittest.TestCase):

    def test_full_and_lora_adaptation_are_compared_from_same_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            summary = run_finetuning_comparison(td, base_steps=18, adaptation_steps=12)
            out_dir = Path(td)

            self.assertTrue((out_dir / "finetuning_summary.json").exists())
            self.assertEqual(
                set(summary["modes"]),
                {"full_no_replay", "full_with_replay", "lora_no_replay", "lora_with_replay"},
            )

            for name, result in summary["modes"].items():
                self.assertAlmostEqual(
                    result["base_loss_before_adaptation"],
                    summary["base_loss_after_base"],
                    places=6,
                    msg=name,
                )
                self.assertAlmostEqual(
                    result["adapted_loss_before_adaptation"],
                    summary["adapted_loss_before_adaptation"],
                    places=6,
                    msg=name,
                )
                self.assertLess(
                    result["adapted_loss_after_adaptation"],
                    summary["adapted_loss_before_adaptation"],
                    name,
                )
                self.assertGreater(result["trainable_parameter_count"], 0, name)
                sample = result["sample"]
                self.assertTrue(
                    all(0 <= idx < summary["vocab_size"] for idx in sample["generated_ids"]),
                    name,
                )

            self.assertGreater(
                summary["modes"]["full_no_replay"]["base_parameter_max_diff"],
                0.0,
            )
            self.assertGreater(
                summary["modes"]["full_with_replay"]["base_parameter_max_diff"],
                0.0,
            )
            self.assertEqual(
                summary["modes"]["lora_no_replay"]["base_parameter_max_diff"],
                0.0,
            )
            self.assertEqual(
                summary["modes"]["lora_with_replay"]["base_parameter_max_diff"],
                0.0,
            )

            full_trainable = summary["modes"]["full_no_replay"]["trainable_parameter_count"]
            lora_trainable = summary["modes"]["lora_no_replay"]["trainable_parameter_count"]
            self.assertLess(lora_trainable, full_trainable)

            expected_checkpoints = {
                "base_model.safetensors",
                "full_no_replay_model.safetensors",
                "full_with_replay_model.safetensors",
                "lora_no_replay_adapter.safetensors",
                "lora_with_replay_adapter.safetensors",
            }
            self.assertTrue(expected_checkpoints.issubset({path.name for path in out_dir.iterdir()}))
            self.assertLess(
                (out_dir / "lora_no_replay_adapter.safetensors").stat().st_size,
                (out_dir / "full_no_replay_model.safetensors").stat().st_size,
            )

    def test_finetuning_comparison_cli_runs(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            result = subprocess.run(
                [
                    sys.executable,
                    "examples/tiny_finetuning_comparison.py",
                    "--out-dir",
                    td,
                    "--base-steps",
                    "3",
                    "--adaptation-steps",
                    "3",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Tiny full vs LoRA comparison completed.", result.stdout)
            self.assertIn("full_no_replay", result.stdout)
            self.assertIn("lora_with_replay", result.stdout)
            self.assertTrue((Path(td) / "finetuning_summary.json").exists())

    def test_cpu_cli_runs_are_exactly_reproducible(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summaries = []
            for run in ("first", "second"):
                out_dir = root / run
                env = os.environ.copy()
                env["PYTHONPATH"] = "."
                subprocess.run(
                    [
                        sys.executable,
                        "examples/tiny_finetuning_comparison.py",
                        "--out-dir",
                        str(out_dir),
                        "--base-steps",
                        "4",
                        "--adaptation-steps",
                        "4",
                        "--device",
                        "cpu",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                summaries.append(
                    json.loads((out_dir / "finetuning_summary.json").read_text(encoding="utf-8"))
                )

            self.assertIn("cpu", summaries[0]["device"].lower())
            keys = ["base_loss_after_base", "adapted_loss_before_adaptation"]
            for key in keys:
                self.assertEqual(summaries[0][key], summaries[1][key])
            for mode in summaries[0]["modes"]:
                for key in ("base_loss_after_adaptation", "adapted_loss_after_adaptation"):
                    self.assertEqual(summaries[0]["modes"][mode][key], summaries[1]["modes"][mode][key])


if __name__ == "__main__":
    unittest.main()
