# 다음 token 예측 해설

1. input은 `[4, 9, 2, 7]`, target은 `[9, 2, 7, 5]`다.
2. `가`의 logit이 가장 높으므로 softmax 이후 확률도 가장 높다.
3. logits는 제한 없는 원시 점수다. probability는 softmax를 거쳐 합이 1이 된 값이다.
4. 학습 문장을 그대로 외워도 loss는 거의 0이 될 수 있다. 보지 않은 validation 또는 test 데이터에서 확인해야 일반화를 말할 수 있다.
5. 다음 token을 정확히 예측하려면 앞 문맥에 포함된 문법, 의미, 사실 관계를 유용한 내부 표현으로 압축해야 하기 때문이다.
6. 문자 단위 tokenizer는 sequence가 길어지고, 자주 함께 쓰이는 단어 조각을 하나의 token으로 표현하지 못한다. 현재 구현은 padded vocabulary와 실제 vocabulary를 별도로 다뤄야 하는 문제도 있다.

핵심은 다음과 같다.

> next-token prediction은 학습 목표이고, 언어 능력은 그 목표를 잘 풀기 위해 생길 수 있는 내부 능력이다. 작은 corpus loss 감소만으로 그 능력이 생겼다고 판단할 수는 없다.

