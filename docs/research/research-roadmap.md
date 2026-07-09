# Mamba-3 경량 적응형 기억 시스템 연구 로드맵

- 문서 상태: 초안 `v0.1`
- 작성일: 2026-07-10
- 실험 브랜치: `exp/cognitive-loop-01`
- 기준 환경: Apple Silicon, MLX

## 1. 연구 목적

이 연구의 1차 목적은 Mamba-3가 Transformer보다 우수하다는 결론을 미리 정해놓고 증명하는 것이 아니다.

검증하려는 중심 질문은 다음과 같다.

> 제한된 로컬 자원에서 `Mamba recurrent state + 외부 episodic memory + 선택적 LoRA consolidation`을 결합하면, 기존 능력을 크게 훼손하지 않으면서 새로운 대화 경험에 적응하는 경량 기억 시스템을 만들 수 있는가?

Mamba-3는 이 시스템의 작업 상태 처리 후보이다. 연구 결과에 따라 Mamba가 적합하지 않다는 결론도 허용한다.

## 2. 세부 연구 질문

1. Mamba recurrent state는 긴 입력 흐름에서 어떤 정보를 얼마나 오래 유지하는가?
2. Mamba state의 크기, 처리 시간, 정확도는 sequence length에 따라 어떻게 변하는가?
3. 새로운 사실을 weight에 학습하지 않고 외부 기억 검색만으로 얼마나 정확히 사용할 수 있는가?
4. 어떤 기억을 외부에만 보존하고, 어떤 반복 패턴을 LoRA 학습 후보로 선별해야 하는가?
5. replay를 포함한 주기적 LoRA 학습이 새 패턴을 익히면서 기존 능력을 보존할 수 있는가?
6. 동일 조건의 작은 Transformer와 비교할 때 Mamba가 메모리, 속도, 긴 흐름 유지에서 실제 이점을 보이는가?

## 3. 현재 연구 가설

아래 내용은 결론이 아니라 반증 가능한 가설이다.

### H1. 작업 상태 가설

Mamba recurrent state는 전체 과거 token을 별도로 보관하지 않고도 현재 작업에 필요한 최근 흐름 일부를 유지할 수 있다.

반증 조건:

- state를 사용한 경로가 같은 입력의 full-context 경로와 허용 오차 안에서 일치하지 않는다.
- sequence가 길어질수록 state 자체의 저장 크기가 계속 증가한다.
- 단순한 장기 의존 synthetic task에서도 유효 정보가 지나치게 빨리 사라진다.

### H2. 외부 기억 가설

변경 가능하고 출처가 필요한 사실은 weight를 바꾸지 않고 외부 episodic memory에 저장하고 검색하는 편이 더 정확하고 안전하다.

반증 조건:

- 통제된 사실 검색에서 필요한 기억을 안정적으로 찾지 못한다.
- 충돌하는 기억의 최신성, 출처, 삭제 요청을 처리하지 못한다.
- 검색 결과를 넣어도 답변 정확도가 개선되지 않는다.

### H3. 선택적 공고화 가설

반복적이고 안정된 행동 패턴만 정제하고 기존 데이터 replay와 함께 LoRA로 학습하면, 모든 대화를 즉시 학습하는 방식보다 망각과 오염을 줄일 수 있다.

반증 조건:

- 여러 설정과 seed에서 새 패턴 성능을 높이면서 기존 성능을 보존하는 구간을 찾지 못한다.
- 외부 기억 검색만 사용하는 방식보다 비용과 오류가 커진다.
- adapter의 검증, 교체, rollback을 안정적으로 수행하지 못한다.

### H4. Mamba 적합성 가설

동일한 parameter 수, 데이터, tokenizer, 학습 예산, 하드웨어 조건에서 Mamba가 작은 Transformer보다 긴 sequence 처리 또는 stateful decode 자원 사용에서 측정 가능한 이점을 보인다.

반증 조건:

- 정확도, peak memory, 처리량을 함께 봤을 때 의미 있는 이점이 없다.
- 이점이 Mamba 구조가 아니라 구현 품질이나 비교 조건 차이로 설명된다.

## 4. 범위

### 포함하는 것

- tiny Mamba-3와 작은 통제 데이터로 수행하는 재현 가능한 실험
- recurrent state, cache, state 저장과 복원 특성
- 한국어 toy corpus와 synthetic memory task
- 외부 episodic memory의 저장, 검색, 충돌, 삭제, 출처 관리
- full fine-tuning, LoRA, replay, 주기적 consolidation 비교
- 동일 조건의 작은 Transformer 기준 모델
- CPU 결정론 기준값과 GPU 반복 측정

### 현재 포함하지 않는 것

- 대규모 foundation model 사전학습
- 인간 뇌의 생물학적 복제
- 모든 대화 turn 직후의 자동 weight 업데이트
- 실제 사용자 데이터를 검토 없이 학습하는 기능
- production 서비스, 다중 사용자 운영, 보안 완성
- Transformer보다 Mamba가 우월하다는 선결론

## 5. 기준 용어

| 용어 | 이 연구에서의 의미 |
|---|---|
| 작업 상태 `Working State` | 현재 입력 흐름을 처리하는 context, Mamba recurrent state, 임시 scratchpad |
| 외부 기억 `Episodic Memory` | 사건, 사실, 출처, 시간, 중요도를 모델 weight 밖에 저장한 명시적 기억 |
| 파라미터 기억 `Parametric Memory` | base weight 또는 LoRA adapter에 압축된 암묵적 패턴 |
| 공고화 `Consolidation` | 외부 기억 중 반복적이고 안정된 패턴을 선별해 학습 데이터로 정제하는 과정 |
| replay | 새 데이터를 학습할 때 기존 대표 데이터를 함께 학습하는 방법 |
| online weight learning | 대화 실행 중 gradient로 weight 또는 adapter를 변경하는 것 |
| state update | 토큰 처리에 따라 recurrent state가 변하는 것. weight 학습과 다름 |

## 6. 현재까지 검증된 기준선

검증된 것:

- MLX Mamba-3의 tiny forward, generation, prefill, step 경로가 실행된다.
- 번들 PyTorch reference와 작은 오차 범위에서 numerical parity가 확인됐다.
- 작은 한국어 고정 corpus를 overfit할 수 있다.
- full checkpoint와 LoRA adapter를 저장하고 다시 불러올 수 있다.
- 새 corpus만 학습하면 기존 corpus loss가 악화되는 망각 현상이 나타났다.
- replay는 현재 toy 조건에서 망각을 줄였다.
- LoRA는 base weight를 보존했지만 기능적 망각을 자동으로 제거하지 않았다.
- CPU에서는 같은 seed의 실험이 재현됐고, GPU에서는 작은 수치 차이가 학습 중 증폭됐다.

아직 검증되지 않은 것:

- 실제 한국어 대화 능력과 일반화
- 실제 pretrained Mamba-3 checkpoint 호환성
- Mamba state의 행동 수준 장기기억 성능
- 외부 기억 저장과 검색이 답변을 개선하는지 여부
- 안전한 기억 선별과 consolidation 정책
- Mamba가 Transformer보다 자원을 적게 쓰거나 더 잘 기억한다는 주장
- 실제 대화 중 안정적인 online weight learning

## 7. 단계별 연구 순서와 통과 조건

### 0단계. 기반 실행과 tiny 학습 검증

상태: 완료

완료 항목:

- source smoke test
- numerical parity와 cache equivalence
- tiny 한국어 overfit과 checkpoint roundtrip
- LoRA adapter roundtrip
- forgetting, replay, full/LoRA 비교
- CPU/GPU 재현성 차이 기록

### 1단계. Mamba recurrent state 특성 검증

목적:

- state가 학습과 다른 임시 작업기억이라는 점을 수치로 확인한다.
- state가 유지할 수 있는 정보와 실패하는 길이를 찾는다.

최소 실험:

- copy, key-value recall, distractor가 포함된 synthetic sequence task
- full prefill, token-by-token state, state reset 조건 비교
- sequence length별 state 크기, peak memory, latency, 정확도 측정
- state 저장 후 복원했을 때 다음 출력 재현 여부 확인

통과 조건:

- full prefill과 stateful 경로가 기존 numerical tolerance를 유지한다.
- state 저장 크기가 sequence length에 따라 증가하지 않는지 측정값으로 확인한다.
- 최소 4개 sequence length에서 정확도와 자원 사용 곡선을 기록한다.
- 성공 길이뿐 아니라 정확도가 무너지기 시작하는 길이를 기록한다.

이 단계가 의미하지 않는 것:

- state가 장기기억이나 weight 학습을 대체한다는 뜻이 아니다.
- Transformer보다 우수하다는 뜻이 아니다.

### 2단계. 외부 episodic memory 검증

목적:

- weight를 바꾸지 않고 사실을 저장, 검색, 수정, 삭제할 수 있는지 확인한다.

최소 실험:

- 100개의 통제된 한국어 사실 저장
- keyword 기준 검색부터 시작
- 동일 주제의 오래된 기억과 최신 기억 충돌
- 출처 표시, 삭제, 검색 제외 검증
- 외부 기억 없음/있음 조건의 답변 정확도 비교

초기 통과 조건:

- 통제된 exact fact의 top-1 검색 정확도 95% 이상
- 최신 기억 선택 정확도 95% 이상
- 삭제된 기억의 재검색 차단 100%
- 모델 weight 변경 없이 결과를 재현

### 3단계. 기억 선별과 consolidation dataset 검증

목적:

- raw 대화를 바로 학습하지 않고, 외부 기억 전용 정보와 LoRA 후보 패턴을 구분한다.

기억 분류:

- 외부 기억 전용: 날짜, 사건, 이름, 출처가 필요한 사실, 민감정보, 변경 가능 정보
- LoRA 후보: 반복적으로 확인된 말투, 안정된 규칙, 지속되는 작업 방식, 일반화 가능한 패턴
- 폐기: 모델의 추측, 일회성 잡음, 검증되지 않은 주장, 중복 내용

통과 조건:

- 최소 50개 기억 후보에 사람이 검토한 기준 label을 만든다.
- LoRA 후보 선별 precision 95% 이상을 우선 목표로 한다.
- 개인정보와 삭제 대상은 LoRA 후보에서 100% 제외한다.
- 각 학습 예제에서 원본 기억과 변환 과정을 추적할 수 있다.

### 4단계. 주기적 LoRA consolidation 검증

목적:

- 정제 데이터와 replay로 기존 능력을 보존하면서 반복 패턴을 adapter에 통합할 수 있는지 확인한다.

최소 비교 조건:

- 외부 기억 검색만 사용
- LoRA, replay 없음
- LoRA, replay 있음
- full fine-tuning, replay 있음
- LoRA learning rate, rank, step의 작은 탐색

초기 통과 조건:

- 독립된 adapted 평가 loss가 학습 전보다 30% 이상 개선된다.
- base 평가 loss 증가는 학습 전 대비 10% 이내다.
- 최소 3개 seed에서 방향이 유지된다.
- adapter 저장, 로드, 비활성화, rollback이 재현된다.

위 조건을 만족하지 못하면:

- LoRA를 장기기억 경로로 채택하지 않는다.
- 외부 기억 검색을 기본으로 유지하고 실패 원인을 기록한다.

### 5단계. 경량 인지 루프 통합

목적:

- 작업 상태, 외부 기억, 선별기, 주기적 consolidation을 하나의 작은 흐름으로 연결한다.

통합 흐름:

```text
사용자 입력
→ Mamba Working State 업데이트
→ 외부 기억 검색
→ 응답 생성
→ 기억 후보 추출
→ 외부 기억 저장 또는 폐기
→ 주기적 검토
→ 통과한 패턴만 LoRA consolidation
→ 평가 통과 시 adapter 채택, 실패 시 rollback
```

통과 조건:

- 통제된 대화 시나리오에서 저장, 검색, 수정, 삭제, 공고화 흐름이 추적된다.
- 외부 기억만으로 해결할 질문이 불필요하게 LoRA 학습으로 넘어가지 않는다.
- adapter 교체 전후의 base/adapted 평가 결과가 자동 기록된다.
- 실패한 adapter가 활성 모델에 남지 않는다.

### 6단계. Mamba와 Transformer 기준 비교

비교 조건:

- parameter 수 차이 5% 이내
- 동일 tokenizer, 데이터 순서, train/eval split
- 동일 optimizer 계열과 학습 예산
- 동일 sequence length와 batch 조건
- 동일 Apple Silicon 장치

측정값:

- validation loss 또는 task accuracy
- peak memory
- 학습 step 시간
- prefill과 decode 처리량
- context/state 저장 크기
- 길이별 기억 정확도
- adaptation 후 base retention

이 비교를 통과하기 전에는 “Mamba가 Transformer보다 실시간 기억에 유리하다”는 문장을 연구 결론으로 사용하지 않는다.

## 8. 실험 공통 규칙

1. 실험 전에 가설, 독립변수, 통제변수, 성공 기준을 기록한다.
2. train, validation, test 데이터를 가능한 한 분리한다.
3. CPU 결과는 결정론 기준값으로 사용한다.
4. GPU 결과는 한 번의 exact loss가 아니라 최소 3회 반복의 평균과 분산으로 기록한다.
5. device, seed, commit, model config, tokenizer, 데이터 checksum을 남긴다.
6. 비교 실험은 한 번에 핵심 변수 하나만 바꾼다.
7. loss와 generation sample을 함께 본다.
8. 실패와 예상 밖 결과를 삭제하지 않는다.
9. 다음 단계 구현은 현재 단계의 통과 조건을 만족한 뒤 시작한다.

## 9. 의사결정 규칙

| 상황 | 기본 결정 |
|---|---|
| 외부 검색만으로 정확히 해결됨 | weight에 학습하지 않음 |
| 자주 바뀌거나 삭제가 필요한 사실 | 외부 기억에만 유지 |
| 반복적이고 안정된 행동 패턴 | consolidation 후보로 검토 |
| 출처가 없거나 모델이 생성한 추측 | 학습 후보에서 제외 |
| LoRA가 base 기준을 초과해 훼손 | adapter 채택 거부 및 rollback |
| Mamba가 기준 Transformer보다 이점 없음 | Mamba 고유 주장 철회 또는 역할 축소 |

## 10. 바로 다음 실험

다음 실험은 인지 루프 전체 구현이 아니다.

`EXP-001`의 질문은 하나로 제한한다.

> Mamba recurrent state는 sequence length가 증가할 때 저장 크기를 일정하게 유지하면서, 통제된 정보 회상 task에서 어느 길이까지 정보를 보존하는가?

이 결과가 나온 뒤 외부 episodic memory 구현으로 넘어간다.

