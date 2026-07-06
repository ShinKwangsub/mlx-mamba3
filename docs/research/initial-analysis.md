# mlx-mamba3 초기 연구 노트

작성일: 2026-07-07

## 현재 상태

- Upstream: https://github.com/Jada42/mlx-mamba3
- 로컬 fork: https://github.com/ShinKwangsub/mlx-mamba3
- Upstream 기준 커밋: `c4e50f5d5f04630fe474c24db28b5387e9424557`
- 패키지 이름: `mlx-mamba-native`
- 범위: Apple Silicon에서 Mamba-3 핵심 아이디어를 MLX로 실행하기 위한 native 구현

## 구현되어 있는 것

- SISO와 MIMO 방식의 Mamba-3 sequence mixer
- `Bx_prev` cache state를 사용하는 exponential-trapezoidal recurrence
- B/C projection에 적용되는 RoPE 스타일 회전 state projection
- Mamba block과 causal attention block을 섞는 hybrid 구조
- prefill, chunked prefill, token-by-token decode cache 경로
- 기본 generation helper
- safetensors 저장/로드 유틸리티
- Mamba mixer의 `in_proj`, `out_proj`에 적용하는 LoRA wrapper

## 지금까지 확인한 것

- GitHub Actions의 최신 macOS test job은 성공했다.
- 로컬에서 Torch가 필요 없는 테스트는 통과했다.
  - `tests.test_step_eq_prefill`
  - `tests.test_hybrid`
  - `tests.test_cache_chunking`
- `.venv`를 만들고 `requirements.txt`를 설치한 뒤 전체 unittest suite가 통과했다.
- `PYTHONPATH=.` 조건에서 toy example들이 동작했다.
  - hybrid generation
  - LoRA fine-tuning demo
  - `safetensors` 설치 후 serialization demo
  - benchmark script

## 초기 우려

- 레포가 매우 젊다. 2026-06-30에 생성됐고, tag와 release가 없으며 contributor도 1명이다.
- numerical parity는 번들된 읽기 쉬운 PyTorch reference 기준이다. 공식 CUDA/Triton 구현이나 실제 공개 checkpoint와 직접 호환된다는 뜻은 아니다.
- `convert_to_lora()`는 모든 layer에 `layer.mixer`가 있다고 가정한다. 그래서 `AttentionBlock`이 들어간 hybrid model에서는 실패한다.
- README는 inference도 `mx.compile`로 compile한다고 설명하지만, 코드에서 명시적으로 `mx.compile`을 호출하는 곳은 training step뿐이다.
- 예제 스크립트는 repo를 설치하거나 `PYTHONPATH=.`를 붙여야 자연스럽게 실행된다.
- 실제 pretrained checkpoint 변환과 로딩 경로는 아직 확립되어 있지 않다.

## 연구 질문

1. 이 구현은 공식 `state-spaces/mamba`의 Mamba-3 config 및 weight layout과 얼마나 가까운가?
2. 실제 Mamba-3 checkpoint를 MLX safetensors로 안정적으로 변환하는 경로를 만들 수 있는가?
3. Apple Silicon에서 성능 병목은 scan, RoPE, projection, attention cache, Python loop 중 어디에 있는가?
4. hybrid model에서 Mamba block만 선택적으로 LoRA 적용하고, 필요하면 attention projection도 LoRA 대상으로 확장할 수 있는가?
5. M1/M2/M3/M4에서 SISO, MIMO, hybrid를 비교하기 위한 최소 benchmark suite는 무엇이어야 하는가?

## 다음 실험 후보

- `convert_to_lora()`가 hybrid config에서 실패하는 문제를 재현하는 regression test 추가
- Mamba block만 선택적으로 감싸도록 LoRA 변환 로직 수정
- MLX model과 번들 PyTorch reference의 parameter name/shape 비교 스크립트 작성
- 여러 sequence length와 MIMO rank에서 `examples/benchmark.py` profile 수집
- tensor shape 흐름과 architecture를 추적하는 작은 notebook 또는 script 작성
