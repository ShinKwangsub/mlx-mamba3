# loss와 학습

## 1. loss는 모델을 움직이는 점수다

언어 모델의 cross-entropy loss는 정답 token에 준 확률을 평가한다. 한 token의 단순한 형태는 다음과 같다.

```text
loss = -log(정답 token 확률)
```

| 정답 확률 | 대략적인 loss |
|---:|---:|
| 0.1 | 2.303 |
| 0.5 | 0.693 |
| 0.9 | 0.105 |

정답 확률이 높아질수록 loss는 작아진다.

## 2. gradient는 수정 방향을 알려준다

모델에는 수많은 parameter가 있다. 각 parameter를 조금 바꿨을 때 loss가 어떻게 달라지는지 계산한 것이 gradient다.

```text
입력
→ forward
→ logits
→ loss
→ backpropagation
→ gradient
→ optimizer update
→ 변경된 parameter
```

gradient 자체가 학습 결과를 저장하는 것은 아니다. optimizer가 gradient를 이용해 parameter를 실제로 변경해야 학습이 일어난다.

## 3. learning rate는 update 크기다

learning rate가 너무 작으면 학습이 매우 느리다. 너무 크면 좋은 지점을 지나치고 loss가 진동하거나 작은 수치 차이가 크게 증폭될 수 있다.

우리 toy 실험은 빠른 확인을 위해 큰 learning rate를 사용했다. 따라서 그 설정을 실제 모델 학습 recipe로 사용하면 안 된다.

## 4. loss가 낮아도 실패할 수 있다

다음 상황은 모두 가능하다.

- training loss는 낮지만 validation loss는 높다.
- 새 데이터 loss는 낮아졌지만 기존 데이터 loss가 높아졌다.
- 평균 loss는 낮지만 중요한 일부 질문에서 실패한다.
- loss는 좋아졌지만 generation sample은 부자연스럽다.

그래서 loss, validation, task accuracy, generation sample, 기존 능력 보존을 함께 봐야 한다.

## 5. 현재 코드와 연결

`mlx_mamba_native/tiny_overfit.py`의 `train_phase()`는 다음 순서를 반복한다.

```text
language_model_loss 계산
→ gradient 계산
→ optimizer.update
→ model parameter와 optimizer state 평가
```

이제 [문제](../problem/readme.md)를 풀어본다.

