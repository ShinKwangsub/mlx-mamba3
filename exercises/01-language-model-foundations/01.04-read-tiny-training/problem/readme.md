# tiny 학습 코드 읽기 문제

## 실행 전 예측

1. replay가 없는 full fine-tuning에서 기존 base corpus loss는 어느 방향으로 움직일 것으로 예상하는가?
2. LoRA 학습 후 `base_parameter_max_diff`는 얼마일 것으로 예상하는가?
3. LoRA의 base parameter가 변하지 않아도 base corpus loss가 변할 수 있는 이유는 무엇인가?
4. full checkpoint와 LoRA adapter 중 어느 파일이 더 작을 것으로 예상하며 그 이유는 무엇인가?

## 실행 후 확인

CLI를 실행하고 아래 표를 채운다.

| 조건 | base loss | adapted loss | trainable parameter 수 | base parameter 최대 변화 |
|---|---:|---:|---:|---:|
| full, replay 없음 | | | | |
| full, replay 있음 | | | | |
| LoRA, replay 없음 | | | | |
| LoRA, replay 있음 | | | | |

5. 실행 전 예측과 다른 결과를 하나 작성하라.
6. 이 실험만으로 “LoRA가 full fine-tuning보다 좋다” 또는 그 반대를 결론 내릴 수 없는 이유를 두 가지 작성하라.
7. 이 실험에서 다음에 하나의 변수만 바꾼다면 무엇을 바꾸고 싶은가? 그 이유는 무엇인가?

작성을 마친 뒤 [해설](../solution/readme.md)을 확인한다.

