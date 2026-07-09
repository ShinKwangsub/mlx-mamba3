# 학습 진행표

이 문서는 진도를 빨리 끝내기 위한 체크리스트가 아니다. 각 항목을 자신의 말로 설명하고 작은 결과를 예측할 수 있을 때 완료 표시한다.

## 0단계. 기초 진단

- [ ] Python의 list, dictionary, function을 구분할 수 있다.
- [ ] tensor shape에서 각 축이 무엇을 뜻하는지 질문할 수 있다.
- [ ] 확률과 평균의 기본 의미를 설명할 수 있다.
- [ ] training과 inference를 구분할 수 있다.

## 1단계. 언어 모델 기초

- [ ] 텍스트가 token ID로 변환되는 과정을 설명할 수 있다.
- [ ] 다음 token 학습에서 input과 target이 한 칸 이동하는 이유를 설명할 수 있다.
- [ ] logits, probability, loss의 관계를 설명할 수 있다.
- [ ] gradient와 optimizer의 역할을 구분할 수 있다.
- [ ] loss 감소와 일반화 성능 향상을 구분할 수 있다.
- [ ] weight, recurrent state, 외부 기억의 차이를 설명할 수 있다.
- [ ] LoRA가 base weight를 보존해도 출력이 달라지는 이유를 설명할 수 있다.
- [ ] tiny 학습 CLI 결과를 읽고 과장되지 않은 결론을 작성할 수 있다.

## 2단계 이후

- [ ] vector와 matrix multiplication을 shape로 추적할 수 있다.
- [ ] Transformer attention과 KV cache를 설명할 수 있다.
- [ ] SSM recurrence와 Mamba state를 설명할 수 있다.
- [ ] 이 저장소의 입력에서 loss까지 호출 흐름을 추적할 수 있다.
- [ ] baseline과 통제변수를 포함한 실험을 설계할 수 있다.
- [ ] 외부 기억과 LoRA consolidation의 역할을 구분해 설계할 수 있다.
- [ ] 자신의 아이디어를 반증 가능한 가설로 작성할 수 있다.

## 한 단원 학습 기록

아래 형식을 개인 연구 노트에 반복해서 사용한다.

```text
날짜:
단원:

오늘 이해한 것:

처음 예상과 달랐던 것:

아직 설명하지 못하는 것:

내가 만든 다음 질문:

다음 실험에서 확인할 것:
```

