# Day 01 Smoke Check

작성일: 2026-07-07

## 목적

첫날 검증의 목적은 크게 가지 않는 것이다. Mamba-3 cognitive loop를 설계하기 전에, fork한 원본 소스가 현재 로컬 환경에서 실제로 실행되는지만 확인한다.

오늘의 질문은 다음 하나다.

> 지금 받은 `mlx-mamba3` 소스는 우리 Apple Silicon + MLX 환경에서 기본 실행이 되는가?

## 실행 환경

- 작업 디렉터리: `/Users/ayajin/PJ/mlx-mamba3-cognitive-loop`
- 브랜치: `exp/cognitive-loop-01`
- Python: `3.12.13`
- MLX: `0.31.2`
- Torch: `2.12.1`
- safetensors: `0.8.0`

## 확인한 것

### 1. 패키지 import

`mlx_mamba_native`의 주요 모듈을 import했고, tiny config로 모델을 만든 뒤 짧은 generation을 실행했다.

결과:

- import 성공
- tiny `MambaLMHeadModel` 생성 성공
- `generate()` 실행 성공

### 2. prefill/step 동등성 테스트

실행:

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_step_eq_prefill
```

결과:

```text
Ran 2 tests
OK
```

확인한 의미:

- full prefill 결과와 token-by-token `step()` 결과가 작은 설정에서 일치한다.
- cache 기반 autoregressive path가 최소한 toy setting에서는 동작한다.

### 3. PyTorch reference numerical parity

실행:

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_numerical
```

결과:

```text
Ran 4 tests
OK
```

확인한 의미:

- SISO/MIMO forward와 step 경로가 번들 PyTorch reference와 작은 오차 범위에서 일치한다.
- 이 검증은 공식 production Mamba-3 checkpoint 호환성을 의미하지는 않는다.

### 4. hybrid generation example

실행:

```bash
PYTHONPATH=. .venv/bin/python examples/generate_hybrid.py
```

결과:

- hybrid Transformer-Mamba-3 model 생성 성공
- token generation 성공

### 5. generation + safetensors save/load example

실행:

```bash
PYTHONPATH=. .venv/bin/python examples/generate.py
```

결과:

- MIMO mode model 생성 성공
- token generation 성공
- `demo_weights.safetensors` 저장 성공
- 새 모델에 weight load 성공
- 원본 모델과 load된 모델의 출력 차이 `0.0`

### 6. toy LoRA fine-tuning example

실행:

```bash
PYTHONPATH=. .venv/bin/python examples/finetune_tinystories.py
```

결과:

- base parameter는 freeze되고 LoRA parameter만 trainable로 잡혔다.
- trainable array는 총 8개였다.
- 50 step 동안 loss가 내려갔다.

```text
Initial Loss: 5.4158
Final Loss:   0.8361
Loss drop:    4.5798
```

확인한 의미:

- LoRA wrapper와 compiled training step이 toy setting에서 동작한다.
- 작은 synthetic dataset에 대해 모델이 loss를 낮출 수 있다.

아직 의미하지 않는 것:

- 실제 TinyStories dataset으로 학습했다는 뜻은 아니다. 현재 예제는 synthetic token data를 사용한다.
- 대화 중 online learning이 가능하다는 뜻도 아니다.
- pretrained Mamba-3 checkpoint를 fine-tuning할 수 있다는 뜻도 아니다.

## 오늘 발견한 작은 주의점

- `mlx` module에는 `__version__` 속성이 없었다. 버전 확인은 `importlib.metadata.version("mlx")`로 해야 한다.
- 예제는 repo를 설치하지 않은 상태에서는 `PYTHONPATH=.`를 붙여 실행하는 것이 안전하다.

## 오늘의 결론

현재 소스는 우리 환경에서 기본적으로 실행된다.

다만 이 코드는 아직 live chat system이 아니다. 현재 확인된 것은 다음 수준이다.

- Mamba-3 구조 구현
- toy generation
- cache 기반 step/prefill path
- PyTorch reference와의 numerical parity
- safetensors 저장/로드
- toy LoRA fine-tuning

아직 없는 것:

- tokenizer
- chat loop
- streaming 응답
- session 관리
- memory 저장/검색
- 실제 pretrained checkpoint
- online learning
- live serving interface

따라서 다음 단계는 cognitive loop 구현이 아니라, 더 작은 검증을 이어가는 것이다.

## 다음 작은 검증 후보

아래 후보들은 `tests/test_tiny_korean_pipeline.py`로 최소 구현과 검증을 완료했다.

1. 아주 작은 한국어 텍스트를 token id로 바꾸는 임시 tokenizer를 만든다.
2. tiny Mamba가 작은 고정 corpus를 overfit할 수 있는지 확인한다.
3. 저장한 checkpoint를 다시 load해서 같은 prompt에 대해 동일한 generation이 나오는지 확인한다.
4. LoRA adapter만 저장/로드할 수 있는지 확인한다.

실행:

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.test_tiny_korean_pipeline -v
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests
```

결과:

```text
tests.test_tiny_korean_pipeline: Ran 3 tests, OK
전체 테스트: Ran 15 tests, OK
```

확인한 의미:

- toy character tokenizer로 한국어 문자열을 token id로 바꾸고 다시 복원할 수 있다.
- 아주 작은 한국어 고정 corpus에 대해 tiny Mamba가 loss를 빠르게 낮출 수 있다.
- 학습된 full checkpoint를 저장/로드한 뒤 logits와 deterministic generation을 재현할 수 있다.
- LoRA adapter parameter만 추출, 저장, 로드할 수 있다.

아직 의미하지 않는 것:

- 이 tokenizer가 실제 한국어 모델 학습에 적합하다는 뜻은 아니다. 문자 단위 toy tokenizer일 뿐이다.
- tiny corpus overfit은 일반화 성능을 의미하지 않는다.
- LoRA adapter 저장/로드가 실시간 학습 안정성을 보장하지 않는다.
- 아직 live chat, memory, routing, cognitive loop는 검증하지 않았다.

다음 작은 검증 후보:

1. toy tokenizer와 tiny overfit을 CLI script로 실행 가능하게 만든다.
2. overfit 전후 generation sample을 사람이 읽을 수 있게 저장한다.
3. 학습 loss curve를 JSONL로 기록한다.
4. tiny corpus를 조금 바꿨을 때 모델이 얼마나 빨리 새 패턴을 외우는지 확인한다.

위 후보들도 `examples/tiny_korean_overfit.py`와 `tests/test_tiny_overfit_artifacts.py`로 최소 구현과 검증을 완료했다.

실행:

```bash
PYTHONPATH=. .venv/bin/python examples/tiny_korean_overfit.py --out-dir /tmp/mlx-mamba3-tiny-korean-overfit
PYTHONPATH=. .venv/bin/python -m unittest tests.test_tiny_overfit_artifacts -v
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests
```

결과:

```text
tests.test_tiny_overfit_artifacts: Ran 2 tests, OK
전체 테스트: Ran 17 tests, OK
```

생성되는 artifact:

- `summary.json`: base/adapted loss와 artifact 경로
- `loss_curve.jsonl`: step별 loss curve
- `samples.md`: 학습 전, base overfit 후, adapted corpus 학습 후 generation sample
- `tokenizer.json`: toy character tokenizer
- `model.safetensors`: 마지막 모델 checkpoint

CLI 실행 예시 결과:

```text
Base loss: 5.2542 -> 0.4343
Adapted loss: 5.3280 -> 0.4473
Base loss after adaptation: 4.1675
```

확인한 의미:

- tiny overfit 검증을 사람이 실행할 수 있는 CLI로 만들었다.
- loss curve를 JSONL로 남길 수 있다.
- generation sample을 사람이 읽을 수 있는 Markdown으로 남길 수 있다.
- corpus를 바꿨을 때 작은 모델이 새 패턴도 빠르게 외울 수 있다.

중요한 한계:

- adapted corpus를 학습한 뒤 base corpus loss가 크게 나빠졌다. 이는 작은 모델에서 새 패턴 학습이 기존 패턴을 방해할 수 있다는 초기 신호다.
- 이것은 online learning 가능성의 증거가 아니라, 오히려 online learning을 조심해야 한다는 경고에 가깝다.
- sample 생성에서는 모델의 padded vocab id가 나오지 않도록 tokenizer vocab 범위로 logits를 제한했다. 이 제한이 없으면 사람이 읽을 수 없는 `<unk>`가 쉽게 나온다.

다음 작은 검증 후보:

1. base corpus를 잊지 않도록 replay를 섞으면 adapted 학습 후 base loss가 덜 망가지는지 확인한다.
2. full fine-tuning과 LoRA adapter fine-tuning의 forgetting 정도를 비교한다.
3. `samples.md`만 보고는 품질 판단이 어려우므로 loss와 sample을 함께 보는 작은 report를 만든다.
4. toy tokenizer 대신 byte-level tokenizer를 붙였을 때 padded vocab 문제와 generation sample이 어떻게 달라지는지 확인한다.

위 1번 후보는 `examples/tiny_replay_comparison.py`와 `tests/test_tiny_replay_comparison.py`로 최소 구현과 검증을 완료했다.

실행:

```bash
PYTHONPATH=. .venv/bin/python examples/tiny_replay_comparison.py --out-dir /tmp/mlx-mamba3-replay-comparison
PYTHONPATH=. .venv/bin/python -m unittest tests.test_tiny_replay_comparison -v
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests
```

결과:

```text
tests.test_tiny_replay_comparison: Ran 2 tests, OK
전체 테스트: Ran 19 tests, OK
```

CLI 실행 예시 결과:

```text
Base loss after base training: 1.1823
No replay base loss after adaptation: 3.9361
With replay base loss after adaptation: 0.5243
No replay adapted loss: 0.4721
With replay adapted loss: 0.6531
```

확인한 의미:

- adapted corpus만 학습하면 base corpus loss가 크게 나빠진다.
- base corpus를 replay로 함께 섞으면 base corpus loss 악화가 크게 줄어든다.
- replay를 섞은 경우에도 adapted corpus loss는 base 학습 직후보다 내려간다.

중요한 한계:

- 이 결과는 아주 작은 toy corpus와 tiny model에서만 확인된 것이다.
- replay가 online learning의 충분한 해법이라는 뜻은 아니다.
- replay를 섞으면 새 corpus만 학습하는 것보다 adapted loss가 약간 높게 남았다. 즉 "덜 잊기"와 "새 패턴만 빠르게 외우기" 사이에 tradeoff가 생길 수 있다.
- 지금은 full fine-tuning 기준이며, LoRA adapter에서 같은 현상이 나타나는지는 아직 검증하지 않았다.

다음 작은 검증 후보:

1. full fine-tuning과 LoRA adapter fine-tuning의 forgetting 정도를 비교한다.
2. replay 비율을 바꿨을 때 base/adapted loss tradeoff가 어떻게 움직이는지 확인한다.
3. loss와 sample을 함께 보여주는 작은 Markdown report를 생성한다.
4. toy tokenizer 대신 byte-level tokenizer를 붙였을 때 padded vocab 문제와 generation sample이 어떻게 달라지는지 확인한다.
