import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_mamba_native.tiny_overfit import run_tiny_korean_overfit


class TestTinyOverfitArtifacts(unittest.TestCase):

    def test_run_writes_loss_samples_and_adaptation_summary(self):
        with tempfile.TemporaryDirectory() as td:
            summary = run_tiny_korean_overfit(td, base_steps=18, adaptation_steps=12)
            out_dir = Path(td)

            expected_files = {
                "loss_curve.jsonl",
                "model.safetensors",
                "samples.md",
                "summary.json",
                "tokenizer.json",
            }
            self.assertTrue(expected_files.issubset({path.name for path in out_dir.iterdir()}))

            loss_rows = [
                json.loads(line)
                for line in (out_dir / "loss_curve.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(loss_rows), 18 + 12 + 2)
            self.assertEqual(loss_rows[0]["phase"], "base")
            self.assertEqual(loss_rows[19]["phase"], "adapted")

            samples = (out_dir / "samples.md").read_text(encoding="utf-8")
            self.assertIn("before_training", samples)
            self.assertIn("after_base_overfit", samples)
            self.assertIn("after_adaptation", samples)
            self.assertIn("광섭은 맘바", samples)

            saved_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["vocab_size"], summary["vocab_size"])
            self.assertLess(summary["base"]["final_loss"], summary["base"]["initial_loss"] * 0.25)
            self.assertLess(summary["adapted"]["final_loss"], summary["adapted"]["initial_loss"] * 0.25)
            self.assertGreater(summary["base_loss_after_adaptation"], summary["base"]["final_loss"])
            for sample in summary["samples"]:
                self.assertTrue(all(0 <= idx < summary["vocab_size"] for idx in sample["generated_ids"]))

    def test_cli_runs_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            result = subprocess.run(
                [
                    sys.executable,
                    "examples/tiny_korean_overfit.py",
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
            self.assertIn("Tiny Korean overfit smoke test completed.", result.stdout)
            self.assertTrue((Path(td) / "summary.json").exists())
            self.assertTrue((Path(td) / "loss_curve.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
