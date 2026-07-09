# loss와 학습 해설

1. 모델 B의 loss가 더 낮다. 정답 확률이 높을수록 `-log(p)`가 작아진다.
2. 아니다. gradient는 방향 정보이고 optimizer가 parameter를 변경해야 weight 학습이 일어난다.
3. training data overfit을 먼저 의심한다. 데이터 분할 오류나 distribution 차이도 확인해야 한다.
4. catastrophic forgetting 또는 학습 간섭의 신호라고 부를 수 있다.
5. 각 step의 작은 parameter 차이가 다음 forward와 gradient를 바꾸고, 큰 update가 이 차이를 반복해서 증폭할 수 있다.
6. 예시는 validation loss, task accuracy, generation sample, base 성능 보존, 여러 seed 반복, peak memory와 latency다.

loss를 읽을 때는 항상 다음 질문을 붙인다.

```text
어느 데이터의 loss인가?
학습에서 본 데이터인가?
비교 기준은 무엇인가?
기존 능력은 유지됐는가?
여러 번 실행해도 같은 방향인가?
```

