# Claude Opus 논문 초안 작성 프롬프트

> **사용 방법**: 새 Claude Opus 세션을 열고 아래 프롬프트를 전체 복사하여 입력하세요.
> `paper_data_summary.md` 파일 내용도 함께 첨부하거나 프롬프트 아래에 붙여넣으세요.

---

## ═══ PROMPT START ═══

당신은 AI/시뮬레이션 분야의 시니어 연구자이자 논문 작성 전문가입니다. 아래에 제공된 실험 데이터와 연구 맥락을 바탕으로 학술 논문 전체 초안을 작성해주세요.

---

### 논문 개요

**제목 (후보):**
- "Can Small Language Models Follow Military Doctrine? Evidence from a Multi-Agent Wargame Simulation"
- "Emergent Tactical Behavior in LLM-Driven Wargame Agents: A Comparative Study with Rule-Based Systems"
- 더 좋은 제목이 있다면 제안해주세요.

**투고 목표 학술지/학회:**
- IEEE Transactions on Games, AAAI Workshop on AI in Games, 또는 유사 SCI/SCIE 저널
- 분량: 8페이지 이내 (IEEE 2-column format 기준)
- 언어: 영어

**핵심 연구 질문:**
- RQ1: LLM 에이전트(Qwen2.5-7B)는 군사 교리 기반 행동(DCR > 0.5)을 자율적으로 수행할 수 있는가?
- RQ3: LLM 에이전트는 규칙 기반 에이전트보다 높은 전술 다양성(행동 엔트로피)을 보이는가?
  (RQ2 모델 간 비교는 데이터 부족으로 제외)

---

### 첨부 데이터

아래 `[DATA SUMMARY]` 섹션에 실험 수치, 통계 결과, 시스템 설계, 한계점이 모두 포함되어 있습니다. 이 데이터를 그대로 사용하여 논문을 작성하세요. 수치를 임의로 수정하거나 추가하지 마세요.

---

### 논문 작성 지침

**필수 포함 항목:**

1. **Abstract** (250단어 이내)
   - 연구 목적, 방법, 핵심 결과(DCR=0.718, p<10⁻⁴⁰; Entropy +40.8%, p<10⁻³⁰), 의의

2. **Introduction**
   - LLM의 에이전트 역할 확대 트렌드
   - 군사 시뮬레이션에서의 의사결정 에이전트 필요성
   - 이 연구의 기여: (a) LLM의 교리 준수 실증, (b) 창발적 전술 행동 분석, (c) 소형 LLM 배포 한계 규명
   - 논문 구조 안내

3. **Related Work**
   - LLM agents in simulation/games (예: Generative Agents, LLM-based game AI)
   - Military wargame simulation AI
   - Lanchester combat models
   - ※ 실제 논문 레퍼런스를 인용하되, 확실하지 않은 출처는 [CITATION NEEDED] 표시

4. **System Architecture**
   - 4.1 Simulation Environment: Hexagonal grid, Lanchester Linear Law 전투 모델, Fog-of-War
   - 4.2 Agent Design: Blue(LLM) / Red(Rule-based) / White Cell(심판)
   - 4.3 Prompting Strategy: FM 3-90 교리 기반 시스템 프롬프트, ASSESS-DEVELOP-DECIDE 구조
   - 4.4 Evaluation Metrics: DCR, Action Entropy, Win Rate, Fallback Rate

5. **Experimental Setup**
   - 5개 시나리오 설명 (표 형식)
   - Phase 3(Rule baseline, n=250) vs Phase 4-1(Qwen LLM, n=100)
   - Fog-of-War 파라미터 (visibility=5, identification=2)

6. **Results**
   - 6.1 RQ1 Doctrine Compliance: 표 + t-test 결과
   - 6.2 RQ3 Action Diversity: Entropy 비교 표 + Mann-Whitney 결과
   - 6.3 Action Type Distribution: 두 에이전트의 행동 분포 비교 (support_by_fire, recon 창발)
   - 6.4 Win Rate Analysis: Fisher's exact, 유의하지 않음을 솔직히 기술

7. **Discussion**
   - LLM이 더 다양한 전술을 사용하지만 승률로 이어지지 않는 이유
   - Fallback 12.12%의 의미와 개선 방향
   - Mistral/Llama 배포 실패 사례 (max-tokens 설정 문제) — 실전 배포 교훈
   - 연구 한계: 단일 모델, Linear Lanchester, 소규모 표본, 인간 전문가 비교 없음

8. **Conclusion**
   - 핵심 기여 3가지 요약
   - 향후 연구 방향 (더 큰 모델, LLM vs LLM, Square Law 모델 등)

---

**문체 및 형식 지침:**

- IEEE 학술 논문 스타일 (수동태 적절히 사용, 객관적 문체)
- 각 섹션은 실제 논문처럼 완성도 있게 작성 (개요나 bullet point 대신 완전한 문장)
- 수식은 LaTeX 형식으로 표기 (예: $DCR = \frac{N_{compliant}}{N_{total}}$)
- 표는 IEEE 스타일로 (Table I, Table II 번호 부여)
- 그림 설명은 "[Figure X: 설명]" 형식으로 자리 표시 (실제 그림 대신)
- 통계 결과는 p-value, effect size, CI 모두 포함

**주의 사항:**

- 제공된 수치(DCR=0.718, p=1.23×10⁻⁴⁰ 등)를 그대로 사용할 것
- Lanchester 모델은 "linear law approximation"으로 정확히 기술 (square law 아님)
- RQ2(모델 비교)는 데이터 미수집을 솔직히 limitation으로 기술
- Mistral/Llama 실패는 "infrastructure challenge"로 framing
- 과도한 주장 금지 — 데이터가 뒷받침하는 범위 내에서만 claim

---

### [DATA SUMMARY]

아래는 실험 데이터 요약입니다. 이 내용을 논문 작성의 기반으로 사용하세요.

---

#### 시뮬레이션 시스템

**환경:**
- Hexagonal grid (axial coordinates)
- Lanchester Linear Law: `blue_loss = red_rate × red_strength × Δt / defense_modifier`
  - α = β = 0.05, stochastic noise σ = 0.1
- Fog-of-War: visibility_radius=5, identification_radius=2 (llm preset)
- 5 terrain types: Open, Mountain, Urban, Forest, River
- Max turns per game: 12
- Action space: {move, attack, hold, support_by_fire, recon, withdraw}

**에이전트:**
- Blue (Phase 3): Rule-based (priority: attack > move toward objective > hold)
- Blue (Phase 4-1): Qwen2.5-7B-Instruct-4bit via MLX backend (Mac M4)
- Red: Rule-based (adversarial, same action space)
- White Cell: LLM-based referee (DCR judgment per turn)

**시스템 프롬프트 구조:**
- FM 3-90 기반 4개 교리 지침
- ASSESS (현재 상황 평가) → DEVELOP (행동 대안 개발) → DECIDE (최적 행동 선택)
- JSON 형식 출력 강제

---

#### 실험 결과 데이터

**Table: Phase 3 Rule Baseline Results (n=250)**

| Scenario | n | Blue Win% | Red Win% | Draw | DCR | Entropy (bits) |
|----------|---|-----------|----------|------|-----|----------------|
| S1 Open Encounter | 50 | 18% | 82% | 0 | 0.658 | 1.242 |
| S2 Mountain Assault | 50 | 8% | 16% | 38 | 0.636 | 1.014 |
| S3 Urban Fight | 50 | 46% | 54% | 0 | 0.572 | 1.684 |
| S4 River Crossing | 50 | 24% | 4% | 36 | 0.541 | 1.095 |
| S5 Breakout | 50 | 36% | 40% | 12 | 0.811 | 1.006 |
| **Overall** | **250** | **26.4%** | **39.2%** | **86** | **0.6438** | **1.2081 ± 0.325** |

**Table: Phase 4-1 Qwen2.5-7B Results (n=100)**

| Scenario | n | Blue Win% | Red Win% | Draw | DCR | Entropy (bits) | Fallback% |
|----------|---|-----------|----------|------|-----|----------------|-----------|
| S1 Open Encounter | 20 | 25% | 75% | 0 | 0.693 | 1.679 | 15.1% |
| S2 Mountain Assault | 20 | 0% | 50% | 10 | 0.761 | 1.774 | 13.5% |
| S3 Urban Fight | 20 | 15% | 85% | 0 | 0.683 | 1.747 | 10.9% |
| S4 River Crossing | 20 | 25% | 50% | 5 | 0.697 | 1.753 | 11.4% |
| S5 Breakout | 20 | 45% | 45% | 2 | 0.758 | 1.550 | 9.7% |
| **Overall** | **100** | **22.0%** | **61.0%** | **17** | **0.7183** | **1.7006 ± 0.204** | **12.12%** |

**Action Type Distribution:**

| Action Type | Rule Blue | Qwen Blue | Δ |
|-------------|-----------|-----------|---|
| move | 60.5% | 49.1% | -11.4%p |
| hold | 28.1% | 27.9% | -0.2%p |
| attack | 3.8% | 9.6% | +5.8%p |
| support_by_fire | 0% | 11.7% | **+11.7%p** |
| recon | 0% | 1.7% | **+1.7%p** |
| withdraw | 7.6% | 0% | -7.6%p |

---

#### 통계 검정 결과

**RQ1 — One-sample t-test (DCR > 0.5 threshold):**
- t-statistic: 22.45
- p-value: 1.23 × 10⁻⁴⁰
- Cohen's d: 2.257 (large effect)
- 95% Confidence Interval: [0.699, 0.738]
- Mean DCR: 0.7183 (n=100 games)

**RQ3 — Mann-Whitney U test (Qwen entropy > Rule entropy):**
- U-statistic: 22,316
- p-value: 8.48 × 10⁻³¹ (one-tailed, alternative='greater')
- |Rank-biserial correlation|: 0.785 (large effect)
- Qwen: 1.7006 ± 0.204 bits
- Rule: 1.2081 ± 0.325 bits
- Relative increase: +40.8%

**Win Rate — Fisher's Exact Test:**
- Qwen Blue: 22/100 (22.0%)
- Rule Blue: 66/250 (26.4%)
- Odds Ratio: 0.786
- p-value: 0.417 (not significant)

---

#### 배포 한계 (Mistral/Llama)

| Model | Backend | Fallback Rate | Root Cause |
|-------|---------|---------------|------------|
| Mistral-7B-Instruct-v0.3 | vLLM | ~99% | Chat template mismatch + max-tokens=4096 |
| Llama-3.1-8B-Instruct | vLLM | 99.6% | max-tokens=4096 → verbose natural language output instead of JSON |
| Qwen2.5-7B-Instruct-4bit | MLX | 12.1% | Functional (used in paper) |

**Llama diagnostic detail**: 499 games completed (100/scenario × 5 scenarios), but 99/100 games per scenario showed 100% fallback. Error: "No JSON object found in model output." Stability test with max-tokens=1024 showed only 8% fallback, confirming max-tokens as the root cause.

---

#### 연구 한계 (Limitations)

1. Single model evaluated (Qwen2.5-7B only) due to deployment failures of other models
2. Linear Lanchester combat model used (planned: Square Law) — does not capture the full combat dynamics described in the original research plan
3. Sample size: 20 seeds/scenario for LLM (vs 50 for baseline) — moderate statistical power
4. Red agent is always rule-based — no LLM vs LLM evaluation
5. No human expert benchmark for comparison
6. Inference time not recorded in JSONL (estimated ~184s/turn from external timing)

---

## ═══ PROMPT END ═══

---

## 보충 지침 (Opus에게 추가로 요청할 내용)

초안 완성 후 다음을 순서대로 요청하세요:

1. **"Related Work 섹션의 실제 레퍼런스를 채워주세요"** — Opus가 알고 있는 실제 논문들로 [CITATION NEEDED] 부분을 채움

2. **"Figure 1: System Architecture 다이어그램을 ASCII/Mermaid로 그려주세요"** — 논문 Fig.1용 구조도

3. **"Table 형식을 LaTeX으로 변환해주세요"** — IEEE 최종 제출용

4. **"논문 전체를 grammarly 수준으로 영문 교정해주세요"** — 최종 polish
