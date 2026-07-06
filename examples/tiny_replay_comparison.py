import argparse
from pathlib import Path

from mlx_mamba_native.tiny_overfit import run_replay_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare tiny adaptation with and without replay.")
    parser.add_argument("--out-dir", default="/tmp/mlx-mamba3-replay-comparison")
    parser.add_argument("--base-steps", type=int, default=30)
    parser.add_argument("--adaptation-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args()

    summary = run_replay_comparison(
        out_dir=Path(args.out_dir),
        base_steps=args.base_steps,
        adaptation_steps=args.adaptation_steps,
        learning_rate=args.learning_rate,
    )

    no_replay = summary["no_replay"]
    with_replay = summary["with_replay"]

    print("Tiny replay comparison completed.")
    print(f"Artifacts: {args.out_dir}")
    print(f"Base loss after base training: {summary['base_loss_after_base']:.4f}")
    print(f"No replay base loss after adaptation: {no_replay['base_loss_after_adaptation']:.4f}")
    print(f"With replay base loss after adaptation: {with_replay['base_loss_after_adaptation']:.4f}")
    print(f"No replay adapted loss: {no_replay['adapted_loss_after_adaptation']:.4f}")
    print(f"With replay adapted loss: {with_replay['adapted_loss_after_adaptation']:.4f}")


if __name__ == "__main__":
    main()
