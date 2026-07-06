import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .model import MambaConfig, MambaLMHeadModel
from .toy_tokenizer import CharTokenizer
from .weights import save_weights


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
