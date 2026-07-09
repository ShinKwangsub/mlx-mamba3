# tiny 학습 코드 읽기

이번 실습의 목표는 모든 수식을 이해하는 것이 아니다. 데이터가 어디에서 만들어지고, 어느 function이 loss를 계산하며, 어떤 parameter가 저장되는지 흐름을 추적하는 것이다.

## 1. 읽을 파일

다음 파일을 순서대로 연다.

```text
mlx_mamba_native/toy_tokenizer.py
mlx_mamba_native/tiny_overfit.py
mlx_mamba_native/train.py
examples/tiny_finetuning_comparison.py
```

## 2. 추적할 흐름

`tiny_overfit.py`에서 다음 이름을 검색한다.

```text
BASE_CORPUS
ADAPTED_CORPUS
make_tiny_config
make_next_token_batch
language_model_loss
train_phase
run_finetuning_comparison
```

전체 흐름은 다음과 같다.

```text
문자열 corpus
→ CharTokenizer
→ token ID
→ input과 target
→ MambaLMHeadModel
→ logits
→ cross-entropy loss
→ gradient
→ optimizer update
→ full checkpoint 또는 LoRA adapter 저장
```

## 3. 직접 실행

실행하기 전에 어떤 조건의 base loss가 가장 나빠질지 먼저 적는다.

```bash
PYTHONPATH=. .venv/bin/python examples/tiny_finetuning_comparison.py \
  --device cpu \
  --out-dir /tmp/mamba-learning-01-04 \
  --base-steps 5 \
  --adaptation-steps 3
```

정확한 수치보다 네 조건의 역할을 본다.

이 명령은 빠른 코드 읽기 실습을 위해 학습 step을 매우 작게 줄였다. 일부 조건에서는 loss가 충분히 내려가지 않거나 replay 조건이 더 나쁘게 나올 수 있다. 이 실행 한 번으로 replay나 LoRA의 효과를 판정하지 않는다.

```text
full_no_replay
full_with_replay
lora_no_replay
lora_with_replay
```

## 4. artifact 확인

다음 파일의 역할을 자신의 말로 적는다.

```text
finetuning_summary.json
base_model.safetensors
full_no_replay_model.safetensors
lora_no_replay_adapter.safetensors
*_loss_curve.jsonl
tokenizer.json
```

이제 [코드 읽기 문제](../problem/readme.md)를 수행한다.
