# 대화 요약과 경량 인지 모듈 연구 방향

작성일: 2026-07-07

## 1. mlx-mamba3를 연구 대상으로 잡은 이유

처음에는 `Jada42/mlx-mamba3`가 무엇인지 확인했다. 이 레포는 Apple Silicon에서 Mamba-3 구조를 MLX로 실험하기 위한 연구용 구현에 가깝다. Transformer처럼 이미 대기업과 큰 생태계가 장악한 방향을 그대로 따라가기보다, 아직 덜 굳어진 Mamba-3 구조를 직접 이해하고 변형해보는 쪽이 개인 연구자에게 더 흥미롭고 가치 있을 수 있다고 보았다.

우리는 이 레포를 `ShinKwangsub/mlx-mamba3`로 fork했고, `/Users/ayajin/PJ/mlx-mamba3`에 로컬 clone을 만들었다. `research/mlx-mamba3` 브랜치에서 연구 노트를 쌓아가기로 했다.

## 2. 언어 모델 학습에 대한 이해

언어 모델은 기본적으로 앞의 token들을 보고 다음 token의 확률 분포를 예측한다. 학습은 실제 정답 token의 확률을 높이는 방향으로 진행된다. 그래서 `loss`가 내려간다는 것은 모델이 정답 token에 더 높은 확률을 주도록 내부 weight가 조정되고 있다는 뜻이다.

다만 이것은 지능의 전부라기보다, 언어에 담긴 인간 사고의 흔적을 압축하는 강력한 학습 게임에 가깝다. 다음 token 예측을 잘하려면 문법, 의미, 상식, 코드 패턴, 추론 형식, 대화 흐름 같은 것들을 함께 배워야 한다.

## 3. 구조와 weight의 관계

학습 결과물은 단순한 숫자 덩어리가 아니라 `구조 + weight + tokenizer + config`의 묶음이다.

Transformer로 학습한 weight는 Transformer 구조의 parameter slot에 들어갈 때 의미가 있고, Mamba로 학습한 weight는 Mamba 구조의 mixer, state projection, recurrence 관련 parameter slot에 들어갈 때 의미가 있다. 따라서 같은 데이터로 학습하더라도 구조가 다르면 weight의 형태와 의미도 달라진다.

시중의 open-weight 모델은 보통 `model.safetensors`만 주는 것이 아니라, `config.json`, `tokenizer.json`, `generation_config.json` 같은 실행에 필요한 최소 구조 정보를 함께 제공한다. 하지만 학습 데이터, 정확한 학습 recipe, optimizer 설정, filtering 방식, RLHF나 distillation 과정은 공개하지 않는 경우가 많다.

## 4. Transformer와 Mamba의 차이

Transformer와 Mamba는 같은 다음 token 예측 목표로 학습할 수 있지만, 내부에서 문맥을 처리하는 구조가 다르다.

- Transformer는 attention을 통해 token들이 서로를 직접 참조한다.
- Mamba 계열은 recurrent state 또는 SSM state를 통해 sequence 정보를 흘려보내며 압축한다.

짧은 sequence와 성숙한 kernel 환경에서는 Transformer가 매우 강하다. 긴 sequence나 decode 추론에서는 Mamba 계열이 구조적으로 유리할 가능성이 있다. 하지만 Mamba라고 해서 자동으로 학습이 더 싸거나 빠른 것은 아니다. 실제 효율은 구현, 하드웨어, sequence length, 모델 크기, kernel 최적화에 따라 달라진다.

## 5. 지능, 의사소통, 학습의 구분

언어는 인간이 의사소통하기 위해 만든 고수준 protocol이다. 하지만 의사소통과 학습은 다르다.

- 의사소통은 내부 상태 일부를 신호로 바꾸어 다른 존재가 감지하게 하는 것이다.
- 학습은 경험이나 신호 때문에 내부 구조가 바뀌고, 다음 예측이나 행동이 달라지는 것이다.

세포는 언어가 아니라 화학 신호, 전기 신호, receptor 접촉, 농도 변화, quorum sensing 같은 방식으로 소통한다. 컴퓨터도 언어 외에 API, binary protocol, embedding vector, sensor data, gradient, weight update 같은 방식으로 신호를 주고받을 수 있다.

현재 LLM과 인간은 주로 언어로 소통하지만, 모델 내부에서는 token id, vector, matrix multiplication, probability distribution으로 변환된다. 프롬프트 대화는 의사소통에 가깝고, training은 weight를 바꾸는 내부 구조 변경에 가깝다.

## 6. 실시간 학습을 위한 구조 변화 필요성

지능을 “변화를 기록하고, 필요한 상황에서 꺼내 쓰고, 업데이트 상황이 오면 변화에 적응하는 능력”으로 본다면, 현재의 거대한 단일 weight 덩어리를 계속 재학습하는 방식은 실시간 적응에 너무 무겁다.

따라서 낮은 리소스로 실시간 적응형 지능을 만들려면 기능을 나누는 구조가 필요할 가능성이 크다.

- 장기 weight: 오래 유지되는 일반 지식과 언어 능력
- 단기 state: 현재 대화와 작업 상태
- episodic memory: 최근 경험과 사건 저장
- adapter 또는 LoRA: 빠르게 업데이트 가능한 작은 학습 층
- consolidation loop: 중요한 경험만 장기 기억으로 천천히 통합

Mamba는 recurrent state를 갖기 때문에 stream 처리, 작업기억, 상태 추적 실험에 잘 어울릴 수 있다. 하지만 Mamba의 inference state가 곧바로 장기 학습을 의미하지는 않는다. 장기 weight를 바꾸려면 여전히 학습 과정이 필요하다.

## 7. 인간 뇌 구조에서 얻은 모듈 아이디어

뇌를 생물학적으로 그대로 복제하기보다, 뇌가 해결한 문제를 기능 단위로 번역하는 접근이 현실적이다.

| 뇌의 기능 | AI 모듈 대응 |
|---|---|
| 해마 | episodic memory, 최근 경험 저장과 검색 |
| 대뇌피질 장기기억 | base model weights, 장기 지식 |
| 전두엽 | planner, executive controller, 목표 관리 |
| 편도체 | risk detector, safety critic, anomaly detector |
| 감각피질 | text, image, audio encoder |
| 작업기억 | context window, recurrent state, scratchpad |
| 기저핵 | action selection, reinforcement learning, habit |
| 소뇌 | 반복 행동 최적화, low-level skill policy |

우리가 처음부터 뇌 전체를 만들 필요는 없다. 작은 cognitive loop를 만들면 된다.

## 8. 아주 가볍게 구현할 수 있는 첫 구조

초기 목표는 “작은 Mamba-3 언어 모델에 몇 가지 인지 모듈을 붙여보는 것”이다.

```text
입력 텍스트
→ Text Encoder / Tokenizer
→ Mamba-3 Tiny LM
→ Working State
→ Episodic Memory 검색
→ Planner가 다음 행동 후보 생성
→ Risk Critic이 위험/모순/불확실성 점검
→ 응답 또는 action
→ 중요한 사건은 Memory에 저장
```

처음에는 전부 거대한 neural module일 필요가 없다. 대부분은 Python class와 작은 규칙, vector search, 간단한 scoring 함수로 시작할 수 있다.

### 최소 모듈

1. `WorkingMemory`
   - 현재 대화의 최근 turn, 목표, 임시 scratchpad를 저장한다.

2. `EpisodicMemory`
   - 사건을 `{time, text, tags, embedding}` 형태로 저장하고 검색한다.

3. `Planner`
   - 현재 입력과 memory 검색 결과를 바탕으로 다음에 할 일을 고른다.

4. `RiskCritic`
   - 위험한 행동, 불확실한 주장, 모순되는 기억을 점검한다.

5. `Consolidator`
   - 반복적으로 등장하거나 중요한 memory를 장기 학습 후보로 표시한다.

6. `MambaCore`
   - 실제 언어 모델 또는 작은 Mamba-3 모델을 감싼다.

## 9. 첫 실험 제안

처음부터 완전한 agent를 만들기보다 다음 정도의 작은 실험이 좋다.

1. 작은 한국어 corpus로 tiny Mamba-3 language model을 학습한다.
2. 학습된 모델을 저장하고 다시 로드해 generation을 확인한다.
3. `EpisodicMemory`를 붙여 최근 대화 사실을 저장하고 검색한다.
4. `RiskCritic`은 처음엔 rule 기반으로 만든다.
5. 새 정보를 weight에 바로 학습하지 않고 memory에 저장했을 때와, LoRA adapter에 짧게 학습했을 때를 비교한다.
6. 이전에 알던 정보를 잊는지, 즉 catastrophic forgetting이 생기는지 확인한다.

이 흐름은 Mamba-3 자체 연구와 “가벼운 뇌 모방형 AI 모듈” 연구를 연결한다.

## 10. 한 줄 요약

우리가 연구하려는 것은 단순히 더 큰 언어 모델을 만드는 것이 아니라, Mamba-3 같은 sequence model을 중심으로 단기 state, episodic memory, planner, risk critic, adapter learning을 나누어 붙이는 작은 인지 아키텍처다. 이것은 거대한 재학습 없이 변화에 적응하는 모델을 만들 수 있는지 실험하기 위한 출발점이다.
