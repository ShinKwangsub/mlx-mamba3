# AI와 Mamba-3 기초 용어집

이 용어집은 이 연구에서 사용하는 의미를 기준으로 한다. 같은 용어가 다른 논문이나 라이브러리에서 조금 다르게 사용될 수 있다.

| 용어 | 쉬운 설명 |
|---|---|
| token | 모델이 한 번에 처리하는 텍스트 단위. 글자, 단어 조각, byte 등이 될 수 있다. |
| tokenizer | 텍스트를 token ID로 바꾸고 다시 텍스트로 복원하는 규칙과 도구다. |
| vocabulary | tokenizer가 구분할 수 있는 token의 전체 목록이다. |
| embedding | token ID를 계산 가능한 vector로 바꾼 표현이다. |
| vector | 여러 숫자를 순서대로 묶은 값이다. 모델은 의미와 상태를 vector로 표현한다. |
| tensor | 다차원 숫자 배열이다. scalar, vector, matrix를 일반화한 개념이다. |
| shape | tensor 각 축의 크기다. 예를 들어 `(2, 16, 32)`는 보통 batch 2, 길이 16, 특징 32를 뜻한다. |
| parameter | 학습으로 변경되는 숫자다. weight와 bias가 대표적이다. |
| weight | 입력을 변환하는 학습 가능한 parameter다. 흔히 모델의 장기 파라미터 기억으로 비유한다. |
| forward | 입력을 모델에 넣어 logits나 예측을 계산하는 과정이다. |
| logits | 확률로 바꾸기 전의 모델 점수다. 값의 합이 1일 필요는 없다. |
| softmax | logits를 합이 1인 확률 분포로 바꾸는 함수다. |
| loss | 모델 예측이 정답에서 얼마나 벗어났는지 나타내는 학습용 숫자다. |
| cross-entropy | 정답 token에 낮은 확률을 줄수록 큰 값을 주는 대표적인 언어 모델 loss다. |
| gradient | loss를 줄이려면 각 parameter를 어느 방향으로 얼마나 움직여야 하는지 알려주는 기울기다. |
| backpropagation | 출력에서 입력 방향으로 gradient를 계산해 전달하는 과정이다. |
| optimizer | gradient를 사용해 parameter를 실제로 업데이트하는 알고리즘이다. Adam이 대표적이다. |
| learning rate | 한 번의 update에서 parameter를 움직이는 크기를 조절한다. 너무 크면 불안정하고 너무 작으면 느리다. |
| training | loss와 gradient를 계산해 parameter를 변경하는 과정이다. |
| inference | 학습된 parameter로 예측하거나 문장을 생성하는 과정이다. 보통 parameter를 변경하지 않는다. |
| overfit | 학습 데이터는 잘 맞히지만 새로운 데이터에서는 잘하지 못하는 상태다. |
| generalization | 학습에서 직접 보지 않은 입력에도 배운 원리를 적용하는 능력이다. |
| checkpoint | 특정 시점의 model weight와 필요한 설정을 저장한 파일 묶음이다. |
| LoRA | base weight는 고정하고 작은 추가 matrix만 학습하는 parameter-efficient fine-tuning 방법이다. |
| adapter | base model에 추가해 특정 행동이나 지식을 조절하는 작은 학습 모듈이다. LoRA도 adapter의 한 종류다. |
| replay | 새 데이터를 학습할 때 기존 대표 데이터를 함께 학습해 망각을 줄이는 방법이다. |
| catastrophic forgetting | 새 내용을 학습한 뒤 기존 능력이 크게 나빠지는 현상이다. |
| context | 현재 예측을 위해 모델에 제공된 token들의 범위다. |
| cache | 이전 계산을 다시 하지 않도록 중간 결과나 state를 보관한 것이다. |
| recurrent state | sequence를 읽으며 계속 갱신되는 제한된 크기의 내부 상태다. weight와 다르다. |
| SSM | State Space Model. 입력 흐름을 state 변화로 표현하는 sequence model 계열이다. |
| attention | 현재 token이 다른 token들의 정보를 가중합으로 직접 참조하는 연산이다. |
| KV cache | Transformer가 autoregressive inference에서 과거 attention key와 value를 저장한 cache다. |
| prefill | prompt 전체를 먼저 처리해 다음 token 생성에 필요한 cache나 state를 만드는 단계다. |
| decode | cache 또는 state를 사용해 다음 token을 하나씩 생성하는 단계다. |
| external memory | model weight 밖의 파일이나 database에 저장해 검색할 수 있는 명시적 기억이다. |
| episodic memory | 언제 어떤 일이 있었는지 사건 중심으로 저장한 외부 기억이다. |
| consolidation | 외부 기억 중 반복적이고 안정된 패턴을 선별해 장기 학습 후보로 정제하는 과정이다. |
| baseline | 새 방법이 실제로 나은지 비교하기 위한 기준 방법이나 모델이다. |
| metric | loss, accuracy, latency, peak memory처럼 결과를 수치로 판단하는 기준이다. |
| seed | 난수 생성의 시작점을 고정하기 위한 값이다. GPU 연산의 완전한 결정론까지 항상 보장하지는 않는다. |

