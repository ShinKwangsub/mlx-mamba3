import os
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_mamba_native.generate import generate
from mlx_mamba_native.model import MambaConfig, MambaLMHeadModel
from mlx_mamba_native.toy_tokenizer import CharTokenizer
from mlx_mamba_native.train import (
    convert_to_lora,
    load_lora_adapters,
    lora_state_dict,
    loss_fn,
    save_lora_adapters,
)
from mlx_mamba_native.weights import load_weights, save_weights


KOREAN_TOY_TEXT = "광섭은 맘바를 연구한다.\n연구 노트는 한국어로 쓴다.\n" * 3


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
    logits = model(inputs)
    loss = nn.losses.cross_entropy(logits, targets)
    return mx.mean(loss)


def train_full_model(model, inputs, targets, steps: int = 30, learning_rate: float = 0.03):
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, language_model_loss)

    initial = language_model_loss(model, inputs, targets).item()
    final = initial
    for _ in range(steps):
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        final = loss.item()
    return float(initial), float(final)


class TestTinyKoreanPipeline(unittest.TestCase):

    def setUp(self):
        mx.random.seed(7)

    def test_char_tokenizer_roundtrip_and_save_load(self):
        tokenizer = CharTokenizer.from_text(KOREAN_TOY_TEXT)
        ids = tokenizer.encode(KOREAN_TOY_TEXT)

        self.assertGreater(tokenizer.vocab_size, 10)
        self.assertEqual(tokenizer.decode(ids), KOREAN_TOY_TEXT)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "toy_tokenizer.json")
            tokenizer.save(path)
            loaded = CharTokenizer.load(path)

        self.assertEqual(loaded.vocab, tokenizer.vocab)
        self.assertEqual(loaded.decode(loaded.encode(KOREAN_TOY_TEXT)), KOREAN_TOY_TEXT)

    def test_tiny_korean_overfit_and_full_checkpoint_roundtrip(self):
        tokenizer = CharTokenizer.from_text(KOREAN_TOY_TEXT)
        inputs, targets = make_next_token_batch(tokenizer, KOREAN_TOY_TEXT)
        config = make_tiny_config(tokenizer.vocab_size)
        model = MambaLMHeadModel(config)

        initial_loss, final_loss = train_full_model(model, inputs, targets)
        self.assertLess(final_loss, initial_loss * 0.2)

        prompt = inputs[:, :8]
        logits_before = model(prompt)
        generated_before = generate(model, prompt, temp=0.0, max_tokens=6)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tiny_korean.safetensors")
            save_weights(model, path)
            loaded_model = MambaLMHeadModel(config)
            load_weights(loaded_model, path)

        logits_after = loaded_model(prompt)
        generated_after = generate(loaded_model, prompt, temp=0.0, max_tokens=6)

        max_diff = np.abs(np.array(logits_before) - np.array(logits_after)).max()
        self.assertLess(max_diff, 1e-6)
        self.assertEqual(generated_after.tolist(), generated_before.tolist())

    def test_lora_adapter_only_save_load_roundtrip(self):
        config = make_tiny_config(vocab_size=32)
        inputs = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 9]])
        model = MambaLMHeadModel(config)

        with tempfile.TemporaryDirectory() as td:
            base_path = os.path.join(td, "base.safetensors")
            adapter_path = os.path.join(td, "lora.safetensors")
            save_weights(model, base_path)

            convert_to_lora(model, r=2, alpha=4.0)
            optimizer = optim.Adam(learning_rate=0.01)
            loss_and_grad = nn.value_and_grad(model, loss_fn)
            for _ in range(5):
                loss, grads = loss_and_grad(model, inputs, targets)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

            adapters = lora_state_dict(model)
            self.assertTrue(adapters)
            self.assertTrue(all(key.endswith((".lora_A", ".lora_B")) for key in adapters))
            self.assertTrue(any(mx.max(mx.abs(value)).item() > 0 for key, value in adapters.items() if key.endswith(".lora_B")))

            save_lora_adapters(model, adapter_path)

            loaded_model = MambaLMHeadModel(config)
            load_weights(loaded_model, base_path)
            convert_to_lora(loaded_model, r=2, alpha=4.0)
            load_lora_adapters(loaded_model, adapter_path)

        probe = mx.array([[1, 2, 3]])
        diff = mx.max(mx.abs(model(probe) - loaded_model(probe))).item()
        self.assertLess(diff, 1e-6)


if __name__ == "__main__":
    unittest.main()
