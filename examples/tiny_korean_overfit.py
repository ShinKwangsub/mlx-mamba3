import argparse
from pathlib import Path

from mlx_mamba_native.tiny_overfit import run_tiny_korean_overfit


def main():
    parser = argparse.ArgumentParser(description="Run a tiny Korean Mamba overfit smoke test.")
    parser.add_argument("--out-dir", default="/tmp/mlx-mamba3-tiny-korean-overfit")
    parser.add_argument("--base-steps", type=int, default=30)
    parser.add_argument("--adaptation-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args()

    summary = run_tiny_korean_overfit(
        out_dir=Path(args.out_dir),
        base_steps=args.base_steps,
        adaptation_steps=args.adaptation_steps,
        learning_rate=args.learning_rate,
    )

    print("Tiny Korean overfit smoke test completed.")
    print(f"Artifacts: {args.out_dir}")
    print(f"Base loss: {summary['base']['initial_loss']:.4f} -> {summary['base']['final_loss']:.4f}")
    print(f"Adapted loss: {summary['adapted']['initial_loss']:.4f} -> {summary['adapted']['final_loss']:.4f}")
    print(f"Base loss after adaptation: {summary['base_loss_after_adaptation']:.4f}")


if __name__ == "__main__":
    main()
