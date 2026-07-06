import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_mamba_native.tiny_overfit import run_replay_comparison


class TestTinyReplayComparison(unittest.TestCase):

    def test_replay_reduces_base_forgetting_in_toy_adaptation(self):
        with tempfile.TemporaryDirectory() as td:
            summary = run_replay_comparison(td, base_steps=18, adaptation_steps=12)
            out_dir = Path(td)

            expected_files = {
                "base_loss_curve.jsonl",
                "base_model.safetensors",
                "no_replay_loss_curve.jsonl",
                "no_replay_model.safetensors",
                "replay_summary.json",
                "tokenizer.json",
                "with_replay_loss_curve.jsonl",
                "with_replay_model.safetensors",
            }
            self.assertTrue(expected_files.issubset({path.name for path in out_dir.iterdir()}))

            no_replay = summary["no_replay"]
            with_replay = summary["with_replay"]

            self.assertLess(no_replay["adapted_loss_after_adaptation"], summary["adapted_loss_before_adaptation"])
            self.assertLess(with_replay["adapted_loss_after_adaptation"], summary["adapted_loss_before_adaptation"])
            self.assertGreater(no_replay["base_loss_after_adaptation"], summary["base_loss_after_base"])
            self.assertLess(
                with_replay["base_loss_after_adaptation"],
                no_replay["base_loss_after_adaptation"] * 0.5,
            )
            saved_summary = json.loads((out_dir / "replay_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["vocab_size"], summary["vocab_size"])
            for mode in ("no_replay", "with_replay"):
                sample = summary[mode]["sample"]
                self.assertTrue(all(0 <= idx < summary["vocab_size"] for idx in sample["generated_ids"]))

    def test_replay_cli_runs(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            result = subprocess.run(
                [
                    sys.executable,
                    "examples/tiny_replay_comparison.py",
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
            self.assertIn("Tiny replay comparison completed.", result.stdout)
            self.assertTrue((Path(td) / "replay_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
