import argparse
from pathlib import Path

import mlx.core as mx

from mlx_mamba_native.tiny_overfit import run_finetuning_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare tiny full and LoRA adaptation with optional replay.")
    parser.add_argument("--out-dir", default="/tmp/mlx-mamba3-finetuning-comparison")
    parser.add_argument("--base-steps", type=int, default=30)
    parser.add_argument("--adaptation-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    summary = run_finetuning_comparison(
        out_dir=Path(args.out_dir),
        base_steps=args.base_steps,
        adaptation_steps=args.adaptation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    print("Tiny full vs LoRA comparison completed.")
    print(f"Artifacts: {args.out_dir}")
    print(f"Device: {summary['device']}")
    print(f"Base loss after base training: {summary['base_loss_after_base']:.4f}")
    print(f"Adapted loss before adaptation: {summary['adapted_loss_before_adaptation']:.4f}")
    for name, result in summary["modes"].items():
        print(
            f"{name}: base={result['base_loss_after_adaptation']:.4f}, "
            f"adapted={result['adapted_loss_after_adaptation']:.4f}, "
            f"trainable={result['trainable_parameter_count']}"
        )


if __name__ == "__main__":
    main()
