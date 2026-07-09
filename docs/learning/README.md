# Mamba-3와 AI 학습 과정

이 과정의 목적은 코드를 따라 실행하는 데서 멈추지 않고, 광섭이 스스로 다음 질문을 만들고 실험을 설계하며 Mamba-3로 구현할 수 있는 것을 상상할 수 있도록 기초 체력을 쌓는 것이다.

연구 진행 문서와 학습 문서는 역할이 다르다.

- `docs/research/`: 우리가 무엇을 검증했고 다음에 무엇을 실험할지 기록한다.
- `docs/learning/`: 그 실험을 이해하고 비판할 수 있도록 개념과 실습을 제공한다.
- `exercises/`: 설명을 읽은 뒤 직접 답하고 해설과 비교하는 공간이다.

## 학습 원칙

1. 공식을 외우기보다 입력이 어떤 shape로 이동하고 무엇이 변하는지 추적한다.
2. 코드를 실행하기 전에 결과를 먼저 예측한다.
3. loss가 내려갔다는 사실과 모델이 유용해졌다는 판단을 구분한다.
4. Mamba의 장점을 가정하지 않고 Transformer 기준 모델과 비교한다.
5. 이해하지 못한 용어를 건너뛰지 않고 [용어집](glossary.md)에 연결한다.
6. AI에게 바로 정답을 묻기 전에 자신의 설명과 예측을 한 문장이라도 작성한다.

## 권장 학습 반복

한 번에 30분에서 60분 정도를 사용한다.

```text
설명 읽기
→ 코드를 실행하기 전 결과 예측
→ 문제에 자신의 말로 답하기
→ 해설과 비교
→ 실제 코드 또는 artifact 확인
→ 틀린 이유를 한 문장으로 기록
→ 다음 질문 만들기
```

답이 틀린 것은 문제가 아니다. 예측 없이 결과만 보는 것이 학습을 느리게 만든다.

## 전체 학습 지도

| 단계 | 학습 내용 | 완료했을 때 할 수 있는 것 | 상태 |
|---|---|---|---|
| 0 | Python, tensor shape, 확률 기초 진단 | 현재 부족한 선수 지식을 구분한다 | 자료 있음 |
| 1 | token, 다음 token 예측, loss, weight, state | tiny 학습 결과를 자신의 말로 설명한다 | 자료 있음 |
| 2 | vector, matrix, gradient, optimizer, MLX | 학습 loop와 parameter update를 추적한다 | 예정 |
| 3 | Transformer와 attention | KV cache와 attention 비용을 설명한다 | 예정 |
| 4 | SSM과 Mamba-1/2/3 | recurrent state와 selective SSM을 설명한다 | 예정 |
| 5 | 이 저장소 코드 읽기 | 입력에서 loss와 generation까지 호출 흐름을 추적한다 | 일부 자료 있음 |
| 6 | 실험 설계와 평가 | baseline, 통제변수, seed, metric을 정한다 | 예정 |
| 7 | 외부 기억과 consolidation | state, 외부 기억, LoRA의 역할을 설계한다 | 예정 |
| 8 | 독자적 구현 구상 | 아이디어를 반증 가능한 가설과 실험으로 바꾼다 | 예정 |

## 현재 시작점

먼저 [기초 진단 문제](../../exercises/00-foundation-check/00.01-current-level/problem/readme.md)를 푼다. 점수는 실력을 평가하기 위한 등급이 아니라 학습 순서를 정하기 위한 지도다.

그다음 아래 순서로 진행한다.

1. [다음 token 예측](../../exercises/01-language-model-foundations/01.01-next-token-game/explainer/readme.md)
2. [loss와 학습](../../exercises/01-language-model-foundations/01.02-loss-and-learning/explainer/readme.md)
3. [weight, state, 외부 기억](../../exercises/01-language-model-foundations/01.03-weights-state-memory/explainer/readme.md)
4. [tiny 학습 코드 읽기](../../exercises/01-language-model-foundations/01.04-read-tiny-training/explainer/readme.md)

학습 상태는 [진행표](progress.md)에 기록한다.

## 질문 수준을 올리는 방법

처음에는 용어를 묻는 질문이 자연스럽다.

```text
token이 무엇인가?
loss가 내려간다는 것은 무엇인가?
```

다음 단계에서는 원인을 묻는다.

```text
왜 LoRA는 base weight를 보존해도 기존 성능을 훼손할 수 있는가?
왜 Mamba state update는 학습이 아닌가?
```

그다음에는 비교 조건을 묻는다.

```text
Mamba와 Transformer의 메모리를 비교하려면 어떤 변수를 같게 해야 하는가?
replay 효과와 LoRA 효과를 분리하려면 어떤 대조군이 필요한가?
```

마지막에는 반증 가능한 구현 질문을 만든다.

```text
Mamba state가 작업기억에 적합하다면 어떤 길이별 recall 곡선이 나와야 하는가?
외부 기억만으로 충분한 정보와 LoRA에 공고화할 패턴을 어떤 metric으로 구분할 수 있는가?
```

우리가 원하는 학습의 결과는 전문용어를 많이 사용하는 것이 아니라, 아이디어를 작은 검증으로 바꾸고 결과가 예상과 다를 때 가설을 수정할 수 있는 능력이다.

