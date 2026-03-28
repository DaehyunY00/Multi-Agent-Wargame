# Paper Data Summary — Multi-Agent Wargame LLM Simulation
> 생성일: 2026-03-28 | Opus 논문 초안 작성용 참고 데이터

---

## 0. 논문화 전략 결정 사항

- **사용 가능 데이터**: Qwen2.5-7B-Instruct-4bit (MLX, Mac M4) 단독
- **제외 모델**: Mistral-7B (vLLM, fallback 99%) / Llama-3.1-8B (vLLM, fallback 99.6%)
  - 제외 원인: `--max-tokens 4096` 설정으로 모델이 JSON 대신 자연어 서술 생성 → "No JSON object found in model output" 오류
  - Mistral은 추가로 chat template 불일치 문제 존재
- **논문 프레임**: "단일 소형 LLM vs 규칙 기반 에이전트" 비교 연구

---

## 1. 실험 설계 요약

### 시뮬레이션 환경
- **격자**: Axial 좌표 헥사곤 (Hex Grid)
- **전투 모델**: Lanchester Linear Law (이산 근사, 1-step)
  - `blue_loss = red_rate × red_strength × Δt / defense_modifier`
  - α = β = 0.05, 확률 노이즈 σ = 0.1
  - **주의**: 연구계획서의 "Square Law"와 실제 구현은 "Linear Law" — 논문에서 정확히 명시 필요
- **Fog of War**: `llm` 프리셋 (visibility_radius=5, identification_radius=2)
- **지형**: 5종 (Open / Mountain / Urban / Forest / River)
- **행동 유형**: move / attack / hold / support_by_fire / recon / withdraw (6종)
- **최대 턴**: 12턴/게임

### 에이전트
| 구분 | Blue Agent | Red Agent |
|------|-----------|-----------|
| **Phase 3 (Baseline)** | Rule-based | Rule-based |
| **Phase 4-1 (LLM)** | Qwen2.5-7B (MLX) | Rule-based |

### 시나리오 (5종)
| ID | 시나리오명 | 특성 |
|----|-----------|------|
| S1 | Open Encounter | 평지 조우전 |
| S2 | Mountain Assault | 산악 공격 |
| S3 | Urban Fight | 도시 전투 |
| S4 | River Crossing | 하천 도하 |
| S5 | Breakout | 돌파 작전 |

### 실험 규모
- **Phase 3 (Rule Baseline)**: 50 seeds × 5 scenarios = **250 게임**
- **Phase 4-1 (Qwen)**: 20 seeds × 5 scenarios = **100 게임**

### LLM 시스템 프롬프트 기반
- FM 3-90 교리 지침 4항 + MDMP 3단계 (ASSESS / DEVELOP / DECIDE)
- White Cell Agent: 턴별 교리 준수 여부 판정 (DCR 계산)

---

## 2. 핵심 실험 결과

### 2-A. Phase 3 Rule-vs-Rule Baseline

| 시나리오 | n | Blue 승 | Red 승 | Draw | DCR | Entropy |
|---------|---|--------|--------|------|-----|---------|
| S1 Open Encounter | 50 | 18% | 82% | 0 | 0.658 | 1.242 |
| S2 Mountain Assault | 50 | 8% | 16% | 38 | 0.636 | 1.014 |
| S3 Urban Fight | 50 | 46% | 54% | 0 | 0.572 | 1.684 |
| S4 River Crossing | 50 | 24% | 4% | 36 | 0.541 | 1.095 |
| S5 Breakout | 50 | 36% | 40% | 12 | 0.811 | 1.006 |
| **전체** | **250** | **26.4%** | **39.2%** | **86** | **0.6438** | **1.2081±0.325** |

**Rule 에이전트 행동 분포 (Blue, S1 기준):**
- move: 60.5%, hold: 28.1%, withdraw: 7.6%, attack: 3.8%
- support_by_fire: 0%, recon: 0% (규칙 기반에 없는 행동)

### 2-B. Phase 4-1 Qwen2.5-7B

| 시나리오 | n | Blue 승 | Red 승 | Draw | DCR | Entropy | Fallback |
|---------|---|--------|--------|------|-----|---------|---------|
| S1 Open Encounter | 20 | 25% | 75% | 0 | 0.693 | 1.679 | 15.1% |
| S2 Mountain Assault | 20 | 0% | 50% | 10 | 0.761 | 1.774 | 13.5% |
| S3 Urban Fight | 20 | 15% | 85% | 0 | 0.683 | 1.747 | 10.9% |
| S4 River Crossing | 20 | 25% | 50% | 5 | 0.697 | 1.753 | 11.4% |
| S5 Breakout | 20 | 45% | 45% | 2 | 0.758 | 1.550 | 9.7% |
| **전체** | **100** | **22.0%** | **61.0%** | **17** | **0.7183** | **1.7006±0.204** | **12.12%** |

**Qwen 에이전트 행동 분포 (Blue, 전체):**
- move: 49.1%, hold: 27.9%, support_by_fire: 11.7%, attack: 9.6%, recon: 1.7%
- withdraw: 0% (Rule에만 존재)
- **Rule 대비 신규 행동**: support_by_fire(+11.7%), recon(+1.7%)

### 2-C. 주요 비교 지표 요약

| 지표 | Rule Baseline | Qwen LLM | 변화 |
|------|--------------|----------|------|
| Blue 승률 | 26.4% | 22.0% | -4.4%p (NS) |
| Red 승률 | 39.2% | 61.0% | +21.8%p |
| Draw 비율 | 34.4% | 17.0% | -17.4%p |
| DCR (평균) | 0.6438 | 0.7183 | **+11.6%** |
| Action Entropy | 1.208 bits | 1.701 bits | **+40.8%** |
| Action 유형 수 | 4종 | 5종 | +1종 |
| Fallback Rate | 0% | 12.12% | — |

---

## 3. 통계 검정 결과

### RQ1: LLM 에이전트의 교리 준수율(DCR)이 0.5를 초과하는가?

- **One-sample t-test**: t = 22.45, **p = 1.23 × 10⁻⁴⁰**
- **Cohen's d**: 2.257 (large effect)
- **95% CI**: [0.699, 0.738]
- **결론**: H₁ 채택 — Qwen LLM은 우연 수준(0.5)을 크게 초과하는 교리 준수율을 보임

### RQ2: 모델 간 전술적 차이 (계획 폐기)

- Mistral 및 Llama 데이터 없음 (99%+ fallback으로 사용 불가)
- 논문에서는 "단일 모델 분석 + 타 모델 배포 한계 기술"로 처리

### RQ3: LLM 에이전트의 행동 다양성이 규칙 기반보다 높은가?

- **Mann-Whitney U test** (Qwen entropy > Rule entropy, 단측):
  - U = 22,316, **p = 8.48 × 10⁻³¹**
  - |r_rb| = 0.785 (large effect)
- **행동 다양성**: Qwen 1.701 bits vs Rule 1.208 bits (+40.8%)
- **결론**: H₃ 채택 — LLM 에이전트는 규칙 기반 대비 유의미하게 높은 행동 다양성 보임

### 추가 검정: 승률 비교 (Fisher's Exact Test)

- Blue 승률: Qwen 22% vs Rule 26.4%
- Odds Ratio = 0.786, p = 0.417 → **유의하지 않음**
- 해석: LLM이 전술적 다양성은 높지만, 순수 전투 승률로의 전환은 제한적

---

## 4. 논문 구조 제안 (IEEE/ACM 스타일, 8페이지 기준)

```
1. Introduction (0.5p)
   - LLM의 의사결정 에이전트로서의 잠재성
   - 군사 교리 준수 + 전술 다양성이 핵심 RQ

2. Related Work (1p)
   - LLM wargame simulation 선행 연구
   - Rule-based vs LLM military agents
   - Lanchester 모델 + hex grid simulation

3. System Design (1.5p)
   3.1 Simulation Architecture (HexGrid, Combat, FoW)
   3.2 Agent Design (Blue LLM / Red Rule / White Cell)
   3.3 Evaluation Metrics (DCR, Entropy, Win Rate)

4. Experimental Setup (0.5p)
   - 5 scenarios × seeds × models
   - Fog-of-War preset (llm)

5. Results (2p)
   5.1 RQ1: Doctrine Compliance (DCR analysis)
   5.2 RQ3: Action Diversity (Entropy analysis)
   5.3 Behavioral Differences (action type distribution)
   5.4 Win Rate Analysis

6. Discussion (1p)
   6.1 Emergent Tactical Behavior
   6.2 LLM Deployment Challenges (Mistral/Llama fallback)
   6.3 Limitations (single model, linear Lanchester, small n)

7. Conclusion (0.5p)
```

---

## 5. 논문 핵심 주장 (Claims)

**Claim 1 (RQ1)**: Qwen2.5-7B는 통계적으로 유의미한 교리 준수율(DCR=0.718, p<10⁻⁴⁰)을 달성하며, 소형 LLM이 군사 교리 지침을 자율적으로 따를 수 있음을 실증한다.

**Claim 2 (RQ3)**: LLM 에이전트는 규칙 기반 에이전트 대비 40.8% 높은 행동 엔트로피를 보이며(p<10⁻³⁰), 훈련된 전술 패턴 외 support_by_fire, recon 등 창발적 행동을 자발적으로 수행한다.

**Claim 3 (Deployment)**: vLLM + max-tokens 과다 설정 시 소형 LLM(Mistral, Llama)의 JSON 출력 실패율이 99%+에 달하며, 실전 배포 시 출력 길이 제어가 핵심 파라미터임을 확인하였다.

---

## 6. 한계 및 향후 연구

- **모델 범위**: 단일 모델(Qwen2.5-7B) — 모델 비교 불가
- **전투 모델**: Linear Lanchester (계획된 Square Law 미달성)
- **표본 크기**: 20 seeds/scenario (통계 파워 제한)
- **적군**: 규칙 기반만 사용 (LLM vs LLM 미검증)
- **인간 전문가 비교**: 없음

**향후 연구**:
- max-tokens ≤ 512 조건에서 Llama/Mistral 재실험
- Square Law 기반 전투 모델로 전환
- GPT-4o / Claude 3.7 등 대형 모델 비교
- 인간 사령관 결정과의 비교 평가

---

## 7. 코드 및 데이터 위치

| 항목 | 경로 |
|------|------|
| 시뮬레이션 코드 | `src/` |
| Phase 3 Rule 결과 | `runs/phase3/baseline/{scenario}/rule-vs-rule/` |
| Phase 4 Qwen 결과 | `runs/phase4/qwen/{scenario}/seed_N.jsonl` |
| 분석 스크립트 | `scripts/evaluate_logs.py`, `run_statistical_tests.py` |
| 검토 보고서 | `연구계획_이행_검토보고서.md` |

**JSONL 구조 (1 line = 1 turn):**
```json
{
  "turn": 3,
  "actions": [{"unit_id": "blue-a", "action_type": "move",
                "metadata": {"fallback": false}, ...}],
  "combat": {"winner": "red", "casualties_by_unit": {...}},
  "state": {"units": {"blue-a": {"strength": 105, ...}}, ...}
}
```

**승패 판정 로직**: 마지막 전투(`combat.winner`)가 있는 턴의 winner 기준; 없으면 최종 전력(strength) 비교 (>110% 우세 시 승리, 그 외 무승부)
