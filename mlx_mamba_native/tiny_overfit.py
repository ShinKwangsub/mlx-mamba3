import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils

from .model import MambaConfig, MambaLMHeadModel
from .toy_tokenizer import CharTokenizer
from .train import convert_to_lora, save_lora_adapters
from .weights import load_weights, save_weights


BASE_CORPUS = "광섭은 맘바를 연구한다.\n연구 노트는 한국어로 쓴다.\n" * 3
ADAPTED_CORPUS = "광섭은 맘바를 실험한다.\n실험 노트는 짧게 쓴다.\n" * 3


def make_tiny_config(vocab_size: int) -> MambaConfig:
    return MambaConfig(
        d_model=32,
        n_layer=1,
        vocab_size=vocab_size,
        ssm_cfg={"d_state": 8, "headdim": 16, "is_mimo": False},
    )


def make_next_token_batch(tokenizer: CharTokenizer, text: str):
    ids = tokenizer.encode(text)
    return mx.array([ids[:-1]]), mx.array([ids[1:]])


def language_model_loss(model, inputs, targets):
    return mx.mean(nn.losses.cross_entropy(model(inputs), targets))


def generation_sample(model, tokenizer: CharTokenizer, prompt: str, max_tokens: int = 12):
    prompt_ids = tokenizer.encode(prompt)
    context = list(prompt_ids)
    generated_ids = []
    for _ in range(max_tokens):
        logits = model(mx.array([context]))
        next_token = mx.argmax(logits[:, -1, :tokenizer.vocab_size], axis=-1).item()
        next_token = int(next_token)
        generated_ids.append(next_token)
        context.append(next_token)
    return {
        "prompt": prompt,
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids),
    }


def write_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, data):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def train_phase(model, optimizer, tokenizer, text: str, phase: str, steps: int, loss_path: Path):
    inputs, targets = make_next_token_batch(tokenizer, text)
    loss_and_grad = nn.value_and_grad(model, language_model_loss)

    initial = float(language_model_loss(model, inputs, targets).item())
    append_jsonl(loss_path, {"phase": phase, "step": 0, "loss": initial})

    final = initial
    for step in range(1, steps + 1):
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        final = float(loss.item())
        append_jsonl(loss_path, {"phase": phase, "step": step, "loss": final})

    return {"phase": phase, "steps": steps, "initial_loss": initial, "final_loss": final}


def scalar_parameter_count(parameters) -> int:
    return sum(value.size for _, value in utils.tree_flatten(parameters))


def parameter_snapshot(model, exclude_lora: bool = False):
    return {
        name: value
        for name, value in utils.tree_flatten(model.parameters())
        if not exclude_lora or not name.endswith((".lora_A", ".lora_B"))
    }


def max_parameter_diff(before: dict, after: dict) -> float:
    if before.keys() != after.keys():
        raise ValueError("Parameter keys changed during training")
    if not before:
        return 0.0
    return max(float(mx.max(mx.abs(before[name] - after[name])).item()) for name in before)


def write_samples(path: Path, samples: list[dict]):
    lines = ["# Tiny Korean Overfit Samples", ""]
    for sample in samples:
        lines.extend([
            f"## {sample['name']}",
            "",
            f"- Prompt: `{sample['prompt']}`",
            f"- Generated ids: `{sample['generated_ids']}`",
            "",
            "```text",
            sample["generated_text"],
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tiny_korean_overfit(
    out_dir: str | Path,
    base_steps: int = 30,
    adaptation_steps: int = 20,
    learning_rate: float = 0.03,
    seed: int = 13,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_path = out_dir / "loss_curve.jsonl"
    loss_path.write_text("", encoding="utf-8")

    tokenizer = CharTokenizer.from_text(BASE_CORPUS + ADAPTED_CORPUS)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    mx.random.seed(seed)
    model = MambaLMHeadModel(make_tiny_config(tokenizer.vocab_size))
    optimizer = optim.Adam(learning_rate=learning_rate)

    samples = [
        {"name": "before_training", **generation_sample(model, tokenizer, "광섭은 맘바")},
    ]

    base_result = train_phase(model, optimizer, tokenizer, BASE_CORPUS, "base", base_steps, loss_path)
    base_loss_after_base = float(language_model_loss(model, *make_next_token_batch(tokenizer, BASE_CORPUS)).item())
    adapted_loss_before_adaptation = float(language_model_loss(model, *make_next_token_batch(tokenizer, ADAPTED_CORPUS)).item())
    samples.append({"name": "after_base_overfit", **generation_sample(model, tokenizer, "광섭은 맘바")})

    adaptation_result = train_phase(model, optimizer, tokenizer, ADAPTED_CORPUS, "adapted", adaptation_steps, loss_path)
    base_loss_after_adaptation = float(language_model_loss(model, *make_next_token_batch(tokenizer, BASE_CORPUS)).item())
    adapted_loss_after_adaptation = float(language_model_loss(model, *make_next_token_batch(tokenizer, ADAPTED_CORPUS)).item())
    samples.append({"name": "after_adaptation", **generation_sample(model, tokenizer, "광섭은 맘바")})

    save_weights(model, str(out_dir / "model.safetensors"))
    write_samples(out_dir / "samples.md", samples)

    summary = {
        "base_steps": base_steps,
        "adaptation_steps": adaptation_steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "vocab_size": tokenizer.vocab_size,
        "base": base_result,
        "adapted": adaptation_result,
        "base_loss_after_base": base_loss_after_base,
        "adapted_loss_before_adaptation": adapted_loss_before_adaptation,
        "base_loss_after_adaptation": base_loss_after_adaptation,
        "adapted_loss_after_adaptation": adapted_loss_after_adaptation,
        "samples": samples,
        "artifacts": {
            "loss_curve": str(loss_path),
            "samples": str(out_dir / "samples.md"),
            "tokenizer": str(out_dir / "tokenizer.json"),
            "model": str(out_dir / "model.safetensors"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def run_replay_comparison(
    out_dir: str | Path,
    base_steps: int = 30,
    adaptation_steps: int = 20,
    learning_rate: float = 0.03,
    seed: int = 13,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharTokenizer.from_text(BASE_CORPUS + ADAPTED_CORPUS)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    mx.random.seed(seed)
    base_model = MambaLMHeadModel(make_tiny_config(tokenizer.vocab_size))
    base_optimizer = optim.Adam(learning_rate=learning_rate)
    base_loss_path = out_dir / "base_loss_curve.jsonl"
    base_loss_path.write_text("", encoding="utf-8")
    base_result = train_phase(base_model, base_optimizer, tokenizer, BASE_CORPUS, "base", base_steps, base_loss_path)

    base_checkpoint = out_dir / "base_model.safetensors"
    save_weights(base_model, str(base_checkpoint))
    base_loss_after_base = float(language_model_loss(base_model, *make_next_token_batch(tokenizer, BASE_CORPUS)).item())
    adapted_loss_before_adaptation = float(language_model_loss(base_model, *make_next_token_batch(tokenizer, ADAPTED_CORPUS)).item())

    modes = {
        "no_replay": ADAPTED_CORPUS,
        "with_replay": BASE_CORPUS + ADAPTED_CORPUS,
    }
    mode_results = {}

    for mode, adaptation_text in modes.items():
        model = MambaLMHeadModel(make_tiny_config(tokenizer.vocab_size))
        load_weights(model, str(base_checkpoint))
        optimizer = optim.Adam(learning_rate=learning_rate)
        loss_path = out_dir / f"{mode}_loss_curve.jsonl"
        loss_path.write_text("", encoding="utf-8")

        train_result = train_phase(model, optimizer, tokenizer, adaptation_text, mode, adaptation_steps, loss_path)
        mode_results[mode] = {
            "train": train_result,
            "base_loss_after_adaptation": float(
                language_model_loss(model, *make_next_token_batch(tokenizer, BASE_CORPUS)).item()
            ),
            "adapted_loss_after_adaptation": float(
                language_model_loss(model, *make_next_token_batch(tokenizer, ADAPTED_CORPUS)).item()
            ),
            "loss_curve": str(loss_path),
            "sample": generation_sample(model, tokenizer, "광섭은 맘바"),
        }
        save_weights(model, str(out_dir / f"{mode}_model.safetensors"))

    summary = {
        "base_steps": base_steps,
        "adaptation_steps": adaptation_steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "vocab_size": tokenizer.vocab_size,
        "base": base_result,
        "base_loss_after_base": base_loss_after_base,
        "adapted_loss_before_adaptation": adapted_loss_before_adaptation,
        "no_replay": mode_results["no_replay"],
        "with_replay": mode_results["with_replay"],
        "artifacts": {
            "base_loss_curve": str(base_loss_path),
            "no_replay_loss_curve": mode_results["no_replay"]["loss_curve"],
            "with_replay_loss_curve": mode_results["with_replay"]["loss_curve"],
            "tokenizer": str(out_dir / "tokenizer.json"),
            "base_model": str(base_checkpoint),
            "no_replay_model": str(out_dir / "no_replay_model.safetensors"),
            "with_replay_model": str(out_dir / "with_replay_model.safetensors"),
        },
    }
    write_json(out_dir / "replay_summary.json", summary)
    return summary


def run_finetuning_comparison(
    out_dir: str | Path,
    base_steps: int = 30,
    adaptation_steps: int = 20,
    learning_rate: float = 0.03,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    seed: int = 13,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharTokenizer.from_text(BASE_CORPUS + ADAPTED_CORPUS)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    mx.random.seed(seed)
    base_model = MambaLMHeadModel(make_tiny_config(tokenizer.vocab_size))
    base_optimizer = optim.Adam(learning_rate=learning_rate)
    base_loss_path = out_dir / "base_loss_curve.jsonl"
    base_loss_path.write_text("", encoding="utf-8")
    base_result = train_phase(base_model, base_optimizer, tokenizer, BASE_CORPUS, "base", base_steps, base_loss_path)

    base_checkpoint = out_dir / "base_model.safetensors"
    save_weights(base_model, str(base_checkpoint))
    base_loss_after_base = float(language_model_loss(base_model, *make_next_token_batch(tokenizer, BASE_CORPUS)).item())
    adapted_loss_before_adaptation = float(
        language_model_loss(base_model, *make_next_token_batch(tokenizer, ADAPTED_CORPUS)).item()
    )

    mode_specs = {
        "full_no_replay": ("full", ADAPTED_CORPUS),
        "full_with_replay": ("full", BASE_CORPUS + ADAPTED_CORPUS),
        "lora_no_replay": ("lora", ADAPTED_CORPUS),
        "lora_with_replay": ("lora", BASE_CORPUS + ADAPTED_CORPUS),
    }
    mode_results = {}

    for mode, (method, adaptation_text) in mode_specs.items():
        mx.random.seed(seed + 1)
        model = MambaLMHeadModel(make_tiny_config(tokenizer.vocab_size))
        load_weights(model, str(base_checkpoint))
        if method == "lora":
            convert_to_lora(model, r=lora_rank, alpha=lora_alpha)

        base_parameters_before = parameter_snapshot(model, exclude_lora=method == "lora")
        base_loss_before_adaptation = float(
            language_model_loss(model, *make_next_token_batch(tokenizer, BASE_CORPUS)).item()
        )
        mode_adapted_loss_before_adaptation = float(
            language_model_loss(model, *make_next_token_batch(tokenizer, ADAPTED_CORPUS)).item()
        )
        trainable_parameter_count = scalar_parameter_count(model.trainable_parameters())
        total_parameter_count = scalar_parameter_count(model.parameters())
        optimizer = optim.Adam(learning_rate=learning_rate)
        loss_path = out_dir / f"{mode}_loss_curve.jsonl"
        loss_path.write_text("", encoding="utf-8")

        train_result = train_phase(model, optimizer, tokenizer, adaptation_text, mode, adaptation_steps, loss_path)
        base_parameters_after = parameter_snapshot(model, exclude_lora=method == "lora")

        if method == "lora":
            checkpoint_path = out_dir / f"{mode}_adapter.safetensors"
            save_lora_adapters(model, str(checkpoint_path))
        else:
            checkpoint_path = out_dir / f"{mode}_model.safetensors"
            save_weights(model, str(checkpoint_path))

        mode_results[mode] = {
            "method": method,
            "replay": mode.endswith("with_replay"),
            "train": train_result,
            "trainable_parameter_count": trainable_parameter_count,
            "total_parameter_count": total_parameter_count,
            "base_parameter_max_diff": max_parameter_diff(base_parameters_before, base_parameters_after),
            "base_loss_before_adaptation": base_loss_before_adaptation,
            "adapted_loss_before_adaptation": mode_adapted_loss_before_adaptation,
            "base_loss_after_adaptation": float(
                language_model_loss(model, *make_next_token_batch(tokenizer, BASE_CORPUS)).item()
            ),
            "adapted_loss_after_adaptation": float(
                language_model_loss(model, *make_next_token_batch(tokenizer, ADAPTED_CORPUS)).item()
            ),
            "loss_curve": str(loss_path),
            "checkpoint": str(checkpoint_path),
            "sample": generation_sample(model, tokenizer, "광섭은 맘바"),
        }

    summary = {
        "base_steps": base_steps,
        "adaptation_steps": adaptation_steps,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "seed": seed,
        "device": str(mx.default_device()),
        "vocab_size": tokenizer.vocab_size,
        "base": base_result,
        "base_loss_after_base": base_loss_after_base,
        "adapted_loss_before_adaptation": adapted_loss_before_adaptation,
        "modes": mode_results,
        "artifacts": {
            "base_loss_curve": str(base_loss_path),
            "tokenizer": str(out_dir / "tokenizer.json"),
            "base_model": str(base_checkpoint),
        },
    }
    write_json(out_dir / "finetuning_summary.json", summary)
    return summary
