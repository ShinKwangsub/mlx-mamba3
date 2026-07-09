# weight, state, 외부 기억

AI 시스템의 기억을 하나의 덩어리로 생각하면 설계가 혼란스러워진다. 이 연구에서는 최소한 네 가지를 구분한다.

| 종류 | 무엇이 저장되는가 | 어떻게 바뀌는가 | 대표 수명 |
|---|---|---|---|
| context | 현재 입력 token | 새 prompt 구성 | 한 요청 또는 제한된 대화 |
| recurrent state | 입력 흐름을 압축한 tensor | token 처리 | session 또는 명시적 reset 전까지 |
| 외부 기억 | 사건, 사실, 출처, 시간 | database write | 수정·삭제 전까지 |
| weight/LoRA | 반복 패턴이 압축된 parameter | gradient 학습 | checkpoint 교체 전까지 |

## 1. state update는 학습이 아니다

Mamba는 token을 읽으며 recurrent state를 갱신한다. 이 변화는 다음 token 계산에 사용되지만 optimizer가 weight를 변경한 것은 아니다.

```text
state update: 입력을 읽는 동안 임시 계산 상태가 변함
weight update: loss와 gradient로 model parameter가 변함
```

## 2. 외부 기억은 약한 기억이 아니다

외부 기억은 모델 안에 들어가지 않았다는 이유로 임시 기억이 아니다. 파일이나 database에 영구 보존할 수 있고 정확한 수정, 삭제, 출처 추적이 가능하다.

날짜, 사람 이름, 프로젝트 상태처럼 자주 바뀌는 사실은 외부 기억에 두는 편이 유리하다.

## 3. LoRA는 패턴 공고화 후보다

LoRA는 base weight를 고정하고 작은 matrix를 추가해 학습한다. 반복되는 말투나 안정된 작업 규칙처럼 매번 검색하기보다 행동 패턴으로 굳히는 정보에 사용할 수 있다.

하지만 LoRA가 base weight를 보존한다고 기존 기능까지 자동으로 보존되는 것은 아니다. adapter가 최종 출력에 개입하기 때문이다.

## 4. 모든 외부 기억을 LoRA로 옮길 필요는 없다

외부 기억은 정확한 사건을 보존하고, LoRA는 여러 사건의 공통 패턴을 압축한다. 둘은 상하 관계가 아니라 다른 역할이다.

```text
외부 기억: 2026년 7월 10일에 어떤 실험을 했는가?
LoRA 후보: 연구 결과를 설명할 때 항상 한계와 반증 조건을 함께 말한다.
```

이제 [분류 문제](../problem/readme.md)를 푼다.

