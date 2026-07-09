# tiny 학습 코드 읽기 해설

1. replay가 없으면 새 corpus에 맞춰 weight가 바뀌면서 기존 base loss가 높아질 가능성이 크다. 현재 toy 실험에서도 이 방향이 관찰됐다.
2. LoRA가 base parameter를 제대로 동결했다면 `0.0`이어야 한다.
3. LoRA adapter가 base layer 출력에 추가되어 최종 logits를 바꾸기 때문이다. 원본 weight 보존과 기능 보존은 같은 개념이 아니다.
4. LoRA adapter가 더 작아야 한다. 전체 model parameter가 아니라 추가 matrix만 저장하기 때문이다.
5. 개인의 예측에 따라 다르다. 중요한 것은 틀린 결과를 지우지 않고 왜 달랐는지 설명하는 것이다.
6. 같은 learning rate와 step 수가 두 방식에 각각 최적인지 알 수 없고, corpus와 model이 너무 작다. 여러 seed와 독립된 평가 데이터도 필요하다.
7. 좋은 후보는 LoRA learning rate, adaptation step, replay 비율, LoRA rank다. 한 번에 하나만 바꿔야 원인을 해석할 수 있다.

실습용 `3` adaptation step 결과가 앞선 기본 `20` step 연구 결과와 다른 방향을 보일 수 있다. 이것은 오류가 아니라 학습이 충분히 진행되지 않았거나 큰 learning rate의 초기 변동이 강한 조건일 수 있다. replay 효과를 판단하려면 step을 고정한 반복 실험과 독립 평가가 필요하다.

artifact의 역할:

- `finetuning_summary.json`: 조건과 최종 측정값을 구조적으로 저장한다.
- `base_model.safetensors`: adaptation 전 공통 출발점이다.
- `full_*_model.safetensors`: 전체 weight가 변경된 model이다.
- `lora_*_adapter.safetensors`: base model과 함께 로드할 추가 parameter다.
- `*_loss_curve.jsonl`: step별 loss 변화를 기록한다.
- `tokenizer.json`: 문자열과 token ID를 같은 규칙으로 변환하기 위해 필요하다.

이 실습의 최종 목표는 숫자를 외우는 것이 아니다. 아래 문장을 자신의 말로 설명할 수 있으면 된다.

> 같은 base checkpoint에서 시작해도 학습 방식과 replay 데이터에 따라 새 내용 적응과 기존 능력 보존의 균형이 달라진다.
