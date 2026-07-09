# 현재 수준 진단 해설

1. `list`는 값의 순서를 중심으로 저장하고 index로 접근한다. `dictionary`는 key와 value의 대응을 저장한다.
2. function은 입력을 받아 계산하고 결과를 반환한다. 모든 function이 반드시 입력이나 반환값을 갖는 것은 아니다.
3. `2 × 4 × 8 = 64`개다.
4. `ids[:-1]`은 `[10, 20, 30]`, `ids[1:]`은 `[20, 30, 40]`이다. 다음 token 학습에서 input과 target을 만들 때 사용한다.
5. function 호출은 현재 parameter로 계산하는 것이다. 학습은 loss, gradient, optimizer를 통해 parameter까지 변경한다.
6. 합은 `1.0`이다. 서로 배타적인 전체 후보의 확률 분포는 합이 1이어야 한다.
7. 평균은 전체적인 중심을 보여주지만 값들의 분산, 극단값, 서로 다른 집단을 숨길 수 있다.
8. loss는 내려간다. 정답에 높은 확률을 줄수록 cross-entropy가 작아진다.
9. gradient는 현재 위치에서 loss가 가장 빠르게 증가하는 방향과 민감도를 나타낸다. optimizer는 보통 그 반대 방향을 이용한다.
10. 최적점을 지나치거나 loss가 진동하고 발산할 수 있다.
11. training은 parameter를 변경한다. inference는 보통 parameter를 고정한 채 예측한다.
12. overfit은 본 학습 데이터를 외우는 것이다. generalization은 보지 않은 데이터에도 배운 패턴을 적용하는 것이다.
13. weight는 gradient 학습으로 오래 유지되는 parameter다. recurrent state는 token을 처리하면서 바뀌는 임시 계산 상태다.
14. LoRA는 base weight를 고정하고 작은 추가 matrix를 학습한다. 최종 출력에는 adapter가 개입하므로 기존 기능이 항상 보존되는 것은 아니다.
15. 외부 메모리는 정확한 수정, 삭제, 출처 추적이 가능하다. 변경 가능하거나 일회성인 사실을 weight에 압축하면 오히려 관리하기 어려워진다.

해설을 읽은 뒤 틀린 질문의 번호만 기록하지 말고, 틀린 이유를 다음 중 하나로 분류한다.

- 용어를 몰랐다.
- 계산 방법을 몰랐다.
- 두 개념을 같은 것으로 생각했다.
- 결과를 너무 크게 일반화했다.

