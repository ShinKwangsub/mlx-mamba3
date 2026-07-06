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
