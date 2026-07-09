# 연구구현 1호: Mamba-Hippocampus + Transformer-Cortex 경량 루프

작성일: 2026-07-07

> 문서 상태: 초기 구현 구상. 전체 연구 순서와 단계별 통과 기준은 `research-roadmap.md`를 우선한다. Mamba state, 외부 기억, LoRA consolidation의 선수 검증을 통과하기 전에는 이 문서의 전체 인지 루프를 한 번에 구현하지 않는다.

## 목적

이 실험은 Mamba-3를 단독 언어 모델로만 보는 대신, 인간 뇌의 기능 분화에서 아이디어를 얻어 작은 인지 루프를 구성할 수 있는지 확인한다.

핵심 가설은 다음과 같다.

- Transformer 계열 모델은 일반 지식, 언어 능력, 깊은 추론을 담당하는 장기 지식 엔진처럼 사용할 수 있다.
- Mamba 계열 모델은 실시간 흐름, recurrent state, 최근 맥락 추적을 담당하는 작업기억 또는 해마 일부 기능처럼 사용할 수 있다.
- 특정 사건과 개인화된 기억은 모델 weight에 바로 넣지 않고 외부 `EpisodicMemory`에 저장하는 것이 더 가볍고 안전하다.
- 위험 판단과 불확실성 점검은 별도 `RiskCritic`으로 분리한다.

## 작업 디렉터리

- 기준 연구 repo: `/Users/ayajin/PJ/mlx-mamba3`
- 실험 worktree: `/Users/ayajin/PJ/mlx-mamba3-cognitive-loop`
- 실험 브랜치: `exp/cognitive-loop-01`

## 첫 구현 범위

처음부터 실제 학습과 대형 모델 연동을 모두 넣지 않는다. 먼저 Python class 기반의 작은 구조를 만든다.

```text
사용자 입력
→ WorkingMemory 업데이트
→ EpisodicMemory 검색
→ Router가 처리 경로 선택
→ MambaStateTracker 또는 CoreModel이 초안 생성
→ 필요하면 TransformerFallback 호출
→ RiskCritic이 검토
→ 응답
→ 중요한 사건을 EpisodicMemory에 저장
```

## 최소 모듈

1. `WorkingMemory`
   - 최근 대화 turn, 현재 목표, 임시 scratchpad를 관리한다.

2. `EpisodicMemory`
   - `{time, text, tags, importance}` 형태의 사건을 저장한다.
   - 첫 버전은 embedding 없이 keyword search로 시작한다.

3. `Router`
   - 단순 규칙으로 처리 경로를 고른다.
   - 예: 불확실성, 계산, 장기 지식 요청이면 fallback 사용.

4. `MambaStateTracker`
   - 첫 버전은 실제 Mamba 모델 대신 최근 흐름 요약을 담당하는 interface로 둔다.
   - 이후 tiny Mamba model 또는 recurrent state 실험을 연결한다.

5. `TransformerFallback`
   - 첫 버전은 stub으로 둔다.
   - 이후 로컬 모델, API, 또는 작은 Transformer baseline으로 교체할 수 있게 한다.

6. `RiskCritic`
   - 위험 단어, 과도한 확신, 기억 충돌을 rule 기반으로 점검한다.

## 성공 기준

- 하나의 CLI demo에서 memory 저장, 검색, router 선택, critic 검토 흐름이 보인다.
- Mamba/Transformer가 실제 모델이 아니어도 module boundary가 명확하다.
- 이후 실제 Mamba-3 tiny model을 붙일 수 있는 interface가 있다.
- 테스트로 핵심 흐름을 고정한다.

## 다음 논의 주제

- 첫 demo를 CLI로 만들지, notebook/script로 만들지 결정한다.
- `EpisodicMemory`를 JSONL 파일로 시작할지 SQLite로 시작할지 정한다.
- Mamba를 처음부터 실제 모델로 붙일지, interface/stub로 먼저 갈지 정한다.
- Transformer fallback을 지금은 stub로 둘지, OpenAI/API/로컬 모델 중 하나로 연결할지 정한다.
