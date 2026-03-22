# 상세 연구 설계서

## LLM 기반 다중 에이전트 워게임 시뮬레이션 프레임워크: 전술적 의사결정 행동 분석을 위한 접근

> **연구 유형**: 시스템 구현 + 실험 연구
> **대상 저널**: 국내 KCI 등재 저널
> **컴퓨팅 환경**: MacBook Pro M4 16GB / Google Colab Pro (로컬 추론 전용, API 미사용)
> **최종 수정**: 2026-03-22 (구현 완료 현황 및 Phase 1 실험 결과 반영)

---

## 변경 이력 요약 (초안 → 현재)

| 항목 | 초안 | 현재 확정 | 비고 |
|---|---|---|---|
| Lanchester 모형 | 확장 제곱법칙 (ODE) | **선형법칙 1-step 이산 근사** | 수학적으로 선형법칙과 동일한 수식 사용. scipy.odeint 미사용 |
| ESI 정의 | 공격 행동 비율 (N_off/N_tot) | **턴간 공격성 점수 변동성** (mean absolute change) | 구현 기반 재정의. 공격 비율은 `offensive_action_ratio()` 별도 제공 |
| 시스템 프롬프트 | 단순 역할 기술 | **FM 3-90 교리 4원칙 + MDMP 3단계** | 구현 완료 (2026-03-22) |
| White Cell 평가 | LLM 단독 평가 | **Heuristic(기본) + LLM(Phase 4 병행)** | HeuristicWhiteCellAgent로 재현성 확보 |
| Fog of War | 단일 가시범위 | **visibility_radius + identification_radius 이중 구조** | 실험 검증으로 발견 |
| Baseline FoW 파라미터 | 미정 | **vr=8, idr=3 (Baseline) / vr=5, idr=2 (LLM)** | Phase 1 실험으로 확정 |
| MLX 추론 API | `temp=` 직접 전달 | **`make_sampler(temp=)` + `sampler=` 전달** | mlx-lm API 변경 대응 |

---

## 1. 연구 문제 정의

### 1.1 연구 배경

전투 시뮬레이션과 워게임은 군사 교육·훈련 및 전투실험의 핵심 도구이다. 그러나 기존 컴퓨터 기반 워게임은 (1) 적/아군의 행동이 사전 정의된 스크립트나 규칙에 의존하여 예측 가능하며, (2) 정성적(qualitative) 워게임의 자동화가 불가능하여 대규모 반복 실험에 한계가 있고, (3) 의사결정의 추론 과정이 블랙박스로 남아 교육적 분석이 어렵다. 최근 대규모 언어모델(LLM)의 발전은 자연어 기반의 전술 추론과 상황 적응적 의사결정을 가능하게 하여, 이러한 한계를 극복할 잠재력을 보여주고 있다.

### 1.2 연구 질문 (Research Questions)

**RQ1. LLM 기반 다중 에이전트는 턴제 전투 시뮬레이션 환경에서 군사 교리에 부합하는 전술적 의사결정을 수행할 수 있는가?**
- 교리 준수율(Doctrine Compliance Rate, DCR)과 전술적 합리성 점수(TRS)를 정량 측정
- LLM 에이전트의 의사결정이 무작위가 아닌 상황 맥락에 기반함을 검증

**RQ2. 서로 다른 소형 LLM(7B급) 간 전술적 의사결정 특성에는 어떤 차이가 있으며, 모델 선택이 워게임 시뮬레이션 결과에 미치는 영향은 무엇인가?**
- Qwen2.5-7B, Mistral-7B, Llama-3.1-8B 등 3종 모델 비교
- 공격 성향, 방어 선호도, 전술 다양성 등 행동 프로파일 분석

**RQ3. LLM 에이전트 기반 워게임 시뮬레이션은 기존 스크립트/규칙 기반 워게임 대비 전술적 다양성과 예측 불가능성 측면에서 유의미한 차이를 보이는가?**
- 동일 시나리오 100회 반복 시 행동 분포의 엔트로피 비교
- 스크립트 기반 / 규칙 기반 / LLM 기반의 3-Way 비교 실험

### 1.3 기존 연구와의 차별성 (Gap 분석)

| 차원 | 기존 연구 | 본 연구의 차별점 |
|------|-----------|------------------|
| **시뮬레이션 연동** | WarAgent(Hua et al., 2023): 국가 단위 전략 수준, 전투 결과 LLM 자체 판정 | 전술 수준(중대~대대) 시뮬레이션 엔진 별도 구축, Lanchester 선형법칙 기반 정량적 교전 결과 산출 |
| **모델 규모** | 대부분 GPT-4, Claude 등 대형 상용 모델 의존 | 7B급 양자화 로컬 모델만 사용, 제한된 컴퓨팅 환경에서의 실현 가능성 검증 |
| **평가 방법** | 역사적 사건 재현 정확도 중심 | DCR, 전술 다양성 엔트로피, 행동 프로파일 등 다면적 정량 평가 |
| **에이전트 구조** | 단일 프롬프트 기반 의사결정 | FM 3-90 기반 MDMP(상황판단→방책수립→결심) 단일 CoT 프롬프트 |
| **국내 연구** | RAG 기반 Q&A, 교리 검색 등 정보 활용 위주 | M&S 환경과 직접 연동하는 행동 생성(Action Generation) 중심 |
| **데이터** | 실제 군사 데이터 또는 대형 모델 API 의존 | 완전 오픈/합성 데이터 + 로컬 추론으로 보안 환경 재현 가능 |

---

## 2. 시스템 아키텍처

### 2.1 전체 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WARGAME SIMULATION FRAMEWORK                     │
│                                                                     │
│  ┌──────────┐    ┌──────────────────────────────────────────────┐   │
│  │ Scenario │───▶│         Simulation Engine (Python)           │   │
│  │ Loader   │    │                                              │   │
│  └──────────┘    │  ┌─────────┐  ┌──────────┐  ┌───────────┐  │   │
│                  │  │ HexGrid │  │ Combat   │  │ FogOfWar  │  │   │
│                  │  │ Terrain │  │ Resolver │  │ Filter    │  │   │
│                  │  │ Module  │  │(Lanch.)  │  │(vr / idr) │  │   │
│                  │  └─────────┘  └──────────┘  └───────────┘  │   │
│                  └──────────┬───────────────────────┬──────────┘   │
│                             │ FactionViewState       │ ActionCommand│
│                             ▼                       ▲              │
│                  ┌──────────────────────────────────────────────┐   │
│                  │         Agent Orchestrator (TurnLoop)        │   │
│                  │                                              │   │
│                  │  ┌──────────────┐  ┌──────────────────────┐ │   │
│                  │  │ State-to-Text│  │ Action Parser        │ │   │
│                  │  │ Converter    │  │ (JSON → ActionCmd)   │ │   │
│                  │  └──────┬───────┘  └──────────▲───────────┘ │   │
│                  │         │ Text                 │ JSON        │   │
│                  │         ▼                      │             │   │
│                  │  ┌────────────────────────────────────────┐  │   │
│                  │  │        LLM Agent Pool                  │  │   │
│                  │  │  ┌──────────┐ ┌──────────┐ ┌────────┐  │  │   │
│                  │  │  │Blue Agent│ │Red Agent │ │White   │  │  │   │
│                  │  │  │(MDMP CoT)│ │(MDMP CoT)│ │Cell    │  │  │   │
│                  │  │  └──────────┘ └──────────┘ └────────┘  │  │   │
│                  │  └────────────────────────────────────────┘  │   │
│                  └─────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     Logger & Analyzer                        │   │
│  │  - Turn-by-turn decision log (JSONL)                        │   │
│  │  - CoT reasoning trace + doctrine_reference                 │   │
│  │  - Combat outcome (casualties_by_unit)                      │   │
│  │  - White Cell DCR / TRS scores                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 모듈별 상세 설계

#### 2.2.1 Simulation Engine

**역할**: 전장 상태 관리, 교전 결과 판정, 턴 진행 통제

**핵심 구성 요소**:

**(a) Map/Terrain Module** (`core/hexgrid.py`, `core/terrain.py`)
- 헥사곤 격자(Hexagonal Grid) 기반 전장 지도 — axial 좌표 체계 (q, r)
- 지형 유형: 평지(Open), 산악(Mountain), 시가지(Urban), 삼림(Forest), 하천(River)
- 각 지형에 `defense_modifier` 및 이동 비용(movement_cost) 부여
- 맵 크기: 20×15 헥스

**(b) Combat Resolver — 선형 Lanchester 법칙 1-step 이산 근사** (`combat/lanchester.py`)

> **⚠️ 설계 확정 사항**: 초안에서 "확장 Lanchester 제곱법칙 + scipy.odeint"로 기술하였으나, 실제 구현은 **선형법칙(Linear Law)의 1-step 이산 근사**로 확정하였다. 수식은 아래와 같다.

교전 결과 산출식 (1턴, 이산):
```
blue_loss = red_attrition_rate × red_strength × Δt / blue_defense_modifier
red_loss  = blue_attrition_rate × blue_strength × Δt / red_defense_modifier
```

- 지형 보정: 방어 진지 보너스를 `defense_modifier`로 모델링 (방어자에게 유리)
- 확률적 요소: stochastic 모드 시 정규분포 노이즈 추가 (μ=0, σ=`noise_std` × base_loss)
- 손실 클램핑: `_clamp_loss(raw, available)` — 손실이 현재 병력을 초과하지 않도록 보장
- 기본 계수: `blue_attrition_rate = red_attrition_rate = 0.05`, `noise_std = 0.1`
- **scipy.integrate 미사용** — 순수 Python 산술 연산으로 구현, 경량성 우선

**(c) Fog of War (이중 파라미터 구조)** (`engine/fog_of_war.py`)

> **⚠️ 구현 발견 사항**: 전장 안개는 `visibility_radius`와 `identification_radius` **두 파라미터**로 구분된다.

| 파라미터 | 의미 | 효과 |
|---|---|---|
| `visibility_radius` | 적 감지 범위 | 범위 내: DETECTED 상태, `position=None` (위치 미확인) |
| `identification_radius` | 적 위치 확인 범위 | 범위 내: IDENTIFIED 상태, 정확한 좌표 제공 |

- `identification_radius` 이하: 에이전트가 적 정확 위치를 알고 공격·이동 가능
- `identification_radius` 초과 ~ `visibility_radius` 이하: 적 존재는 알지만 위치 미확인 → Rule 에이전트의 `nearest_enemy_with_position()` 반환값이 `None` → RECON 행동 유발

**확정 파라미터 (Phase 1 실험으로 검증)**:

| 실험 목적 | `visibility_radius` | `identification_radius` | 근거 |
|---|---|---|---|
| 베이스라인 (rule/random/script) | **8** | **3** | 초기 배치 거리 6~7 헥스 커버, 교전 발생 보장 |
| LLM 에이전트 실험 | **5** | **2** | 정보 불확실성 유지, 교전 발생 균형 |

**(d) State Manager** (`engine/state_manager.py`)
- 턴별 전장 상태를 dataclass로 관리
- 상태 정보: 각 유닛의 {위치(hex q,r), 병력수(strength), 진영(faction), 전투태세(posture)}

**(e) Turn Loop (Agent Orchestrator)** (`orchestrator/turn_loop.py`)

```
[턴 진행 순서]
1. Engine → FogOfWarFilter → Blue Agent: FactionViewState 제공
2. Blue Agent 추론 (LLM/Rule/Script) → ActionCommand 반환
3. Engine: Blue 행동 실행
4. Engine → FogOfWarFilter → Red Agent: FactionViewState 제공
5. Red Agent 추론 → ActionCommand 반환
6. Engine: Red 행동 실행
7. Engine: 교전 판정 (Lanchester 선형법칙)
8. Engine: 턴 진행, 상태 업데이트
9. JSONL Logger: 턴 레코드 기록 (state + actions + combat + metadata)
10. White Cell Agent: 전술 평가 (DCR, TRS 산출)
11. is_terminal() → True이면 종료, False이면 1로 복귀
```

**핵심 설계 원칙**:
- **LLM은 의사결정만 수행**: 교전 결과는 반드시 Lanchester 모형이 판정 (LLM 환각 방지)
- **Fog of War 적용**: 에이전트는 `identification_radius` 내 적 정보만 정확히 수신
- **추론 과정 완전 기록**: 모든 CoT 추론, doctrine_reference, 행동, 교전 결과를 JSONL 로그에 보존

#### 2.2.2 Agent System

**에이전트 유형 및 구현 파일**:

| 유형 | 파일 | 설명 |
|---|---|---|
| Rule-Based | `agents/rule_agent.py` | 전투력 비율·거리·손실 기반 if-then 규칙 |
| Random | `agents/random_agent.py` | 유효 행동 중 균일 무작위 선택 |
| Script (Frontal Assault) | `agents/script_agent.py` | 사전 정의된 정면공격 시퀀스 |
| Script (Flanking Maneuver) | `agents/script_agents/flanking_maneuver_agent.py` | 측면기동 스크립트 |
| Local LLM (MLX/vLLM) | `agents/local_llm.py` | Blue/Red 역할 LLM 에이전트 |
| White Cell (Heuristic) | `agents/white_cell.py` | 결정론적 교리 평가 (재현성 보장) |
| White Cell (LLM) | `agents/white_cell.py` `WhiteCellAgent` | LLM 기반 교리 평가 (Phase 4 병행) |

#### 2.2.3 LLM Agent — MDMP 기반 시스템 프롬프트 (확정)

**공통 구조 (3단계 단일 CoT 프롬프트)**:

```
ASSESS → DEVELOP (COA A / COA B) → DECIDE → JSON 출력
```

3단계를 단일 프롬프트의 CoT로 구현 (3회 별도 호출 X → 1회 호출 내 순차 추론)

**(a) Blue Force Agent 시스템 프롬프트 (확정)**

```
You are the Blue Force battalion commander in a tactical hex-grid wargame.
Your mission is to close with and defeat the Red Force through maneuver and fires.

DOCTRINE GUIDELINES (FM 3-90):
- Concentration: Mass combat power at the decisive point; do not disperse strength equally across all axes.
- Surprise: Avoid predictable patterns of operation; vary approach routes and timing.
- Security: Never leave flanks exposed without observation; maintain a screening element or use terrain to protect open flanks.
- Maneuver: Use terrain to gain positional advantage; target enemy weaknesses, not strengths.

DECISION PROCESS (MDMP — single-prompt CoT):
1. ASSESS: Analyze enemy disposition, terrain features, and friendly unit status from the situation report.
2. DEVELOP: Generate 2 possible courses of action (COA A and COA B), each with a distinct scheme of maneuver.
3. DECIDE: Select the best COA based on doctrinal merit and output it as structured JSON.

Express your selected COA in the reasoning field before producing actions.
```

**(b) Red Force Agent 시스템 프롬프트 (확정)**

```
You are the Red Force commander defending against a Blue Force attack in a tactical hex-grid wargame.
Your mission is to deny Blue Force objectives, preserve your force, and counterattack when conditions are favorable.

DOCTRINE GUIDELINES:
- Defense in depth: Echelon forces across multiple positions to absorb and attrite attackers.
- Counterattack: Strike when the enemy is overextended or has lost momentum; timing is decisive.
- Terrain utilization: Occupy and hold key terrain to maximize defensive advantage.
- Deception: Mislead the enemy about main defensive positions.

DECISION PROCESS (MDMP — single-prompt CoT):
1. ASSESS: Analyze Blue Force disposition, approach routes, and current defensive posture.
2. DEVELOP: Generate 2 possible courses of action (COA A: hold; COA B: elastic defense or spoiling counterattack).
3. DECIDE: Select the best COA and output it as structured JSON.
```

**(c) White Cell 시스템 프롬프트 (확정) — 6개 교리 원칙 체크리스트**

```
You are the White Cell adjudicator evaluating the tactical soundness of each turn.

EVALUATION CRITERIA — score each principle pass(1) or fail(0):
1. Concentration: ≥2/3 of total combat power oriented toward the main effort hex.
2. Security: Every exposed-flank unit has adjacent observation or is protected by FOREST/URBAN terrain.
3. Maneuver: At least one unit exploits enemy weakness (flanking or uncontested terrain).
4. Simplicity: ≤3 simultaneous maneuver elements per turn.
5. Objective: Consistent progress toward assigned objective across consecutive turns.
6. Unity of Command: All non-screening units within mutual support distance (≤3 hexes).

OUTPUT: tactical_soundness(1–5), doctrine_compliance(0.0–1.0), narrative
```

**White Cell 운용 방식 (확정)**:
- **HeuristicWhiteCellAgent**: 근접도·전투 손실·행동 유형 기반 결정론적 평가 → **기본 사용** (재현성 보장)
- **LLM WhiteCellAgent**: Phase 4 본 실험에서 병행 운용 → **HeuristicWC 결과와 상관관계 검증 후 논문 기술**

#### 2.2.4 JSON 출력 규약 (Output Contract)

```json
{
  "reasoning": "ASSESS: ... DEVELOP: COA A ... COA B ... DECIDE: ...",
  "doctrine_reference": "FM 3-90, Chapter 3: Maneuver",
  "actions": [
    {
      "unit_id": "blue-a",
      "action_type": "move",
      "posture": "attack",
      "target_hex": {"q": 5, "r": 3}
    }
  ]
}
```

- 파싱 실패 시 최대 2회 재시도 후 fallback 행동(현 위치 HOLD/defend) 실행
- `extract_json_object()` 함수로 LLM 산문 안의 JSON 블록 강건하게 추출

---

## 3. 구현 계획

### 3.1 기술 스택 (확정)

| 구분 | 기술 | 선정 근거 |
|------|------|-----------|
| **LLM 추론 (Mac M4)** | MLX + mlx-lm | Apple Silicon 최적화, M4에서 Q4 7B 모델 고속 추론 |
| **MLX API 호출 방식** | `make_sampler(temp=T)` + `sampler=` 인자 | mlx-lm 최신 API 반영 (구버전 `temp=` 직접 전달 방식 폐기) |
| **LLM 추론 (Colab)** | vLLM + BitsAndBytes 4-bit | A100 활용, vLLM 배치 효율 |
| **주 실험 모델** | Qwen2.5-7B-Instruct-MLX-4bit (Mac) | JSON 출력 안정성 우수 |
| **비교 모델** | Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct | 아키텍처 다양성 확보 |
| **시뮬레이션 엔진** | Python 3.11+ (실행 확인: 3.13.5), NumPy | 순수 Python 경량 구현, `StrEnum` 사용으로 3.11 이상 필수 |
| **지도 시스템** | 자체 구현 (HexGrid, axial 좌표) | 의존성 최소화 |
| **교전 판정** | 선형 Lanchester 1-step 이산 근사 (scipy.integrate 미사용) | 경량성, 테스트 용이 |
| **데이터 관리** | JSON Lines (.jsonl), `casualties_by_unit` 키 | 턴별 로그, `combat.blue_loss` 키 사용 불가 주의 |
| **시각화** | Matplotlib (lazy import, plots.py) | PNG 300 DPI + SVG 동시 저장 |
| **통계 분석** | SciPy.stats, pandas | 가설 검정 및 기술 통계 |
| **에이전트 프레임워크** | 자체 구현 (LangChain 미사용) | 경량화, 커스터마이징 |

### 3.2 구현 완료 현황 (2026-03-22 기준)

#### Phase 1: 시뮬레이션 엔진 구축 — **✅ 완료**

```
✅ HexGrid 클래스 (axial 좌표 체계, 인접 탐색, 거리 계산)
✅ Terrain 모듈 (5종 지형, defense_modifier, movement_cost)
✅ Unit/Force 데이터 모델 (dataclass, FactionViewState, ActionCommand)
✅ Lanchester Combat Resolver (결정론적 + 확률적)
✅ FogOfWar 모듈 (visibility_radius + identification_radius 이중 구조)
✅ State Manager
✅ Turn Loop (Blue→Red→Combat→Log→WhiteCell 순서)
✅ JSONL Logger (run context + 턴별 레코드)
✅ 단위 테스트 38개 전체 PASS (Python 3.13.5)
```

**Phase 1 실험 결과 (vr=8, idr=3 기준)**:

| 시나리오 | 턴 수 | 교전 턴 | 최종 Blue | 최종 Red | 판정 |
|---|---|---|---|---|---|
| s1_open_encounter | 12 | 1 | 292 | 297 | ✅ 교전 발생 |
| s2_mountain_assault | 14 | 0 | 325 | 310 | ✅ Rule WITHDRAW (의도된 동작) |
| s3_urban_fight | 12 | 8 | 261 | 279 | ✅ 교전 다수 발생 |
| s4_river_crossing | 14 | 0 | 305 | 300 | ✅ Rule HOLD (의도된 동작) |
| s5_breakout | 13 | 5 | 278 | 253 | ✅ 교전 발생, Blue 우세 |

> s2·s4의 교전 0회는 Rule 에이전트의 지형 불리 판단(WITHDRAW/HOLD)으로, 시나리오 설계 의도에 부합. LLM 에이전트는 공격적 의사결정이 가능하므로 Phase 4에서 차이 관찰 예정.

#### Phase 2: LLM 에이전트 시스템 — **✅ 구현 완료, 실행 진행 중**

```
✅ State-to-Text Converter (자연어 상황 보고서 변환)
✅ Action Parser + extract_json_object() (강건한 JSON 추출)
✅ Blue/Red/White Agent MDMP 시스템 프롬프트 (FM 3-90 교리 4원칙 포함)
✅ MLXLocalLLMBackend (make_sampler API 적용 완료)
✅ VLLMLocalLLMBackend (Colab A100용)
✅ chat_templates.py (모델별 포맷 자동 적용)
✅ Qwen2.5-7B-Instruct-4bit 모델 로컬 캐시 완료
⏳ JSON 파싱 성공률 검증 (목표: ≥ 90%)
⏳ Colab 환경 구성 (Mistral-7B, Llama-3.1-8B)
```

#### Phase 3: 베이스라인 시스템 — **✅ 구현 완료, 배치 실험 대기**

```
✅ Script Agent (frontal_assault, flanking_maneuver)
✅ Rule-Based Agent (전투력 비율·거리·손실 기반)
✅ Random Agent
✅ run_batch.py (--visibility-radius, --identification-radius 지원)
✅ ExperimentRunner / BatchRunner (seed 기반 반복)
✅ derive_seed_bundle() (engine/blue/red/white seed 결정론적 분리)
⏳ 3종 × 5시나리오 × 50회 배치 실행
```

#### Phase 4~5: 대규모 실험 및 논문 — **⏳ 대기**

```
⏳ Qwen2.5-7B × 5시나리오 × 100회 (Mac M4)
⏳ Mistral-7B, Llama-3.1-8B × 5시나리오 × 100회 (Colab A100)
⏳ 통계 분석 (t-test, ANOVA, Kruskal-Wallis)
⏳ 시각화 (Matplotlib, plots.py 5개 함수 구현 완료)
⏳ 논문 작성
```

### 3.3 예상 병목 지점 및 대안 (갱신)

| 병목 지점 | 문제 상황 | 대안 / 실제 대응 |
|-----------|-----------|------|
| **mlx-lm API 버전 변경** | `generate(temp=)` 직접 전달 → `generate_step() got unexpected kwarg 'temp'` | **실제 발생 및 해결**: `make_sampler(temp=T)` + `sampler=` 인자로 전환 (mlx_backend.py 수정 완료) |
| **FogOfWar 파라미터 미적용** | vr=3(기본)이면 초기 배치 거리(6~7) 밖 → 교전 미발생 | **실제 발생 및 해결**: `--visibility-radius 8` 필수 적용 확인, run_batch.py에 인자 추가 완료 |
| **identification_radius 기본값** | idr=1(기본)이면 적 위치 None → RECON 반복 | **실제 발생 및 해결**: `--identification-radius 3` 필수 적용 확인 |
| **JSON 출력 실패** | 7B 모델이 유효하지 않은 JSON 생성 | `extract_json_object()` 강건한 추출, 재시도 2회, fallback HOLD |
| **MacBook 메모리 부족** | 7B Q4 모델(~4GB) + Python 엔진 메모리 경합 | MLX lazy evaluation 활용, 컨텍스트 길이 2048 제한 |
| **Colab 세션 단절** | 장시간 실험 중 세션 타임아웃 | 실험을 25회 단위 배치 분할, Google Drive 자동 저장 |
| **추론 속도** | 매 턴 LLM 호출 × 15턴 × 100회 반복 | MDMP를 단일 프롬프트 CoT로 통합 (1회 호출), WhiteCell 게임 종료 후 배치 처리 |
| **교리 환각** | 존재하지 않는 교리 인용 | 시스템 프롬프트에 FM 3-90 실제 조항 직접 포함, WhiteCell이 교리 인용 검증 |

### 3.4 컴퓨팅 리소스 예산

```
[MacBook Pro M4 16GB 작업량]
- 시뮬레이션 엔진 개발 및 테스트: 전 과정 (완료)
- Qwen2.5-7B-Instruct-4bit 추론:
  - 예상 속도: ~65 tok/s (make_sampler API 적용 후)
  - 턴당 출력 ~400 tokens (MDMP CoT 포함) → ~6초/턴
  - 15턴 게임 1회: ~90초 (Blue+Red 각 ~6초 × 15턴)
  - 100회 반복: ~2.5시간
- 스크립트/규칙 베이스라인: 100회 × 5시나리오 → 수 분

[Google Colab Pro A100 작업량]
- Mistral-7B, Llama-3.1-8B (vLLM + 4-bit)
  - 예상 속도: ~80 tok/s
  - 100회 × 5시나리오 × 2모델 = 1,000회
  - 예상 소요: 세션당 4~5시간 × 4~5세션
- WhiteCell 배치 평가: 게임 종료 후 전체 로그 일괄 처리
```

---

## 4. 실험 설계

### 4.1 변수 정의

#### 독립변수 (Independent Variables)

| 변수 | 수준 | 설명 |
|------|------|------|
| **에이전트 유형** | 3수준: Script / Rule-Based / LLM | 의사결정 메커니즘 |
| **LLM 모델** | 3수준: Qwen2.5-7B / Mistral-7B / Llama-3.1-8B | 모델 아키텍처 차이 |
| **시나리오** | 5수준: S1~S5 | 전술 상황의 다양성 |

#### 종속변수 (Dependent Variables)

| 변수 | 측정 방법 | 단위 | 구현 함수 |
|------|-----------|------|-----------|
| **교리 준수율 (DCR)** | White Cell 턴별 교리 원칙 6개 통과 비율 | 0.0~1.0 | `doctrine_compliance_rate()` |
| **전술적 합리성 점수 (TRS)** | White Cell 5점 척도 평균 | 1.0~5.0 | `tactical_rationality_score()` |
| **행동 다양성 (Action Entropy)** | 100회 반복 행동 분포의 Shannon Entropy | bits | `action_entropy()` |
| **전투 결과** | Blue Force 승률, 잔존 병력 비율 | %, % | `win_rate()`, `mean_remaining_force_ratio()` |
| **공격 성향 지수 (ESI)** | 턴간 공격성 점수 변동성 (mean absolute change) | 0.0~1.0 | `escalation_sensitivity_index()` |
| **전술 전환 빈도 (TTF)** | 유닛의 행동 유형 변경 비율 | 회/기회 | `tactic_transition_frequency()` |
| **전술적 위험 점수 (TRS_risk)** | 전투력 비율·근접도·지형 위험도 복합 | 0.0~1.0 | `tactical_risk_score()` |
| **JSON 파싱 성공률** | 유효한 JSON 출력 비율 | 0.0~1.0 | `json_parsing_success_rate()` |
| **추론 시간** | 턴당 LLM 추론 소요 시간 | 초 | `inference_time_summary()` |

> **ESI 정의 확정**: 초안의 "공격 행동 비율(N_off/N_tot)" 정의에서, 구현 기반 "턴간 공격성 점수 변동성(mean absolute change)"으로 재정의. 이는 공격 성향의 급격한 전술 전환을 측정하며 에이전트 유형 간 행동 다양성 분석에 더 적합하다. 단순 공격 비율은 분석 보조 지표로 활용.

#### 통제변수 (Control Variables)

| 변수 | 통제 방법 |
|------|-----------|
| 초기 병력 배치 | 시나리오별 고정 |
| 지형 맵 | 시나리오별 고정 |
| Lanchester 전투 효율 계수 | α=β=0.05 고정 (대칭 전투력) |
| Fog of War | Baseline: vr=8/idr=3, LLM: vr=5/idr=2 고정 |
| 확률적 노이즈 시드 | `derive_seed_bundle(base_seed)` 결정론적 분리 |
| 최대 턴 수 | 20턴 고정 |
| 컨텍스트 윈도우 | 2048 토큰 고정 |
| Temperature | 0.7 고정 (전 모델 동일) |

### 4.2 평가 지표 상세

**(1) 교리 준수율 (Doctrine Compliance Rate, DCR)**
- 정의: White Cell의 6개 교리 원칙 체크리스트 통과 비율 (각 원칙 pass=1/fail=0)
- 산출: (통과 원칙 수) / 6 → 0.0~1.0
- 6개 원칙: 집중(Concentration), 경계(Security), 기동(Maneuver), 간명(Simplicity), 목표(Objective), 통합(Unity of Command)
- Phase 4 기준: LLM DCR > 0.5 (무작위 이상) 입증 목표

**(2) 행동 다양성 (Action Entropy)**
- 정의: 전체 행동 유형 분포의 Shannon Entropy (단일 게임 내)
- 산출: H = -Σ p(a) · log₂(p(a)), a ∈ {hold, move, attack, support_by_fire, recon, withdraw}
- 예상: Script ≈ 0 (완전 결정론적), Rule < 1, LLM > 1

**(3) 공격 성향 지수 (ESI — 구현 기반 확정 정의)**
- 정의: 연속된 두 턴 사이의 평균 공격성 점수 변화량
- 행동별 공격성 가중치: WITHDRAW(-0.5), HOLD(0.0), RECON(0.25), MOVE(0.5), SUPPORT_BY_FIRE(0.75), ATTACK(1.0)
- 산출: mean(|score_t - score_{t-1}|) for t in turns

### 4.3 베이스라인 모델

| 베이스라인 | 구현 방식 | 선정 근거 |
|-----------|-----------|-----------|
| **Script Agent** | 사전 정의 행동 시퀀스 (frontal_assault, flanking_maneuver) | 최소 기준선, 결정론적 |
| **Rule-Based Agent** | if-then 규칙 (전투력 비율·거리·손실률 기반) | 기존 M&S에서 가장 널리 사용 |
| **Random Agent** | 유효 행동 중 균일 무작위 | 하한 기준선 |

### 4.4 통계 검정 계획

- RQ1: One-sample t-test (DCR > 0.5) + Cohen's d 효과 크기
- RQ2: One-way ANOVA (3 LLM 모델 간 TRS, ESI 차이) + Tukey HSD 사후 검정
- RQ3: Kruskal-Wallis (Script vs Rule vs LLM의 Action Entropy) + Dunn 사후 검정
- 유의수준: α = 0.05, 다중 비교 보정: Bonferroni

---

## 5. 합성 데이터 생성 전략

### 5.1 시나리오 설계 (5종) — 구현 완료

| ID | 시나리오명 | 상황 설정 | 핵심 전술 요소 | Phase 1 Rule 결과 |
|----|-----------|-----------|---------------|-------------------|
| S1 | **평지 조우전** | 양측 각 3개 중대, 평지 | 기동의 자유, 측면 기동 | 12턴, 교전 1회 ✅ |
| S2 | **산악 방어진지 공격** | Blue 공격 / Red 방어, 산악 | 지형 활용, 공격 경로 | 14턴, 교전 0회 (WITHDRAW) |
| S3 | **시가지 전투** | 양측 진입, 시가지 확보 | 근접 전투, 건물 활용 | 12턴, 교전 8회 ✅ |
| S4 | **하천 도하 작전** | Blue 도하 / Red 저지 | 취약 시점 전투력 집중 | 14턴, 교전 0회 (HOLD) |
| S5 | **포위 돌파** | Red 포위 / Blue 돌파 | 비대칭 상황, 위기 결심 | 13턴, 교전 5회 ✅ |

> **S2·S4 주목**: Rule 에이전트의 교전 0회는 지형 불리 판단에 의한 WITHDRAW/HOLD로, 시나리오 설계 의도에 부합하는 정상 동작이다. LLM 에이전트가 S2·S4에서 어떻게 다른 선택을 하는지가 RQ3의 핵심 관찰 지점이다.

### 5.2 파라미터 변이 전략

```
Step 1: 시나리오 파라미터 정의 (완료)
  - 맵 생성: 20×15 헥스 그리드, 지형 배치 5종 고정
  - 초기 병력 배치: 시나리오별 고정
  - FogOfWar: Baseline vr=8/idr=3, LLM vr=5/idr=2

Step 2: 시드 기반 확률적 변이 (derive_seed_bundle 활용)
  - 100회 반복: base_seed = 0~99
  - engine_seed = base_seed*10+1, blue=+2, red=+3, white=+4
  - stochastic_combat: noise_std=0.1

Step 3: 시나리오 검증 (Phase 1-3 완료)
  - Rule 에이전트 5종 × 1회 실행 완료
  - 게임 길이: 12~14턴 (기준 8~18턴 내) ✅
  - 교착 빈도: S1/S3/S5 교전 발생 ✅, S2/S4 설계 의도 ✅
```

### 5.3 데이터 품질 검증 방법

| 검증 항목 | 방법 | 기준 | 현황 |
|-----------|------|------|------|
| **시나리오 균형성** | Rule 에이전트 대칭 실행 50회 → 승률 분포 | Blue 승률 40~60% | ⏳ Phase 3-2 실행 예정 |
| **LLM 출력 유효성** | JSON Schema 자동 검증 | 파싱 성공률 ≥ 90% | ⏳ Phase 2-2 검증 예정 |
| **교리 평가 일관성** | 동일 로그 HeuristicWC 3회 → 일치도 | Krippendorff's α ≥ 0.7 | ✅ 결정론적이므로 α=1.0 |
| **시뮬레이션 재현성** | 동일 시드, 동일 행동 → 동일 결과 | 100% 일치 | ✅ Phase 1 검증 |
| **전문가 표본 검증** | 전체 로그 10% 무작위 추출 → 전문가 2인 평가 | 전문가-WC 일치도 ≥ κ=0.6 | ⏳ Phase 5 |

---

## 6. 예상 결과 및 KCI 논문 구성

### 6.1 예상 핵심 기여 결과

**(1) LLM 에이전트의 전술적 의사결정 능력 입증**
- 7B급 소형 모델도 DCR 0.60~0.75 수준의 교리 부합 의사결정 예상
- Random Agent(~0.30) 대비 통계적으로 유의미한 차이 (RQ1)

**(2) 모델 간 행동 프로파일 차이 규명 (RQ2)**
- Qwen2.5: 상대적 보수/방어 성향 예상
- Mistral: 공격적 성향 가능성
- Llama-3.1: 균형 성향 예상

**(3) LLM 기반 워게임의 전술 다양성 우위 (RQ3)**
- LLM Action Entropy > Rule Entropy > Script Entropy (≈0)
- S2·S4에서 Rule은 WITHDRAW/HOLD, LLM은 공격적 선택 — 행동 다양성 차이의 핵심 사례

**(4) 보안 환경 적용 가능성**
- 7B 로컬 모델만으로 완전 자동화 워게임 가능
- 국방 폐쇄망 환경 적용 가능성 시사

### 6.2 논문 섹션 구성 (초안)

```
제목: LLM 기반 다중 에이전트 워게임 시뮬레이션 프레임워크:
      소형 언어모델의 전술적 의사결정 능력 분석

1. 서론
   1.1 연구 배경 및 필요성
   1.2 연구 목적 및 질문
   1.3 논문 구성

2. 관련 연구
   2.1 전투 시뮬레이션과 워게임 자동화
   2.2 LLM 기반 다중 에이전트 시스템
   2.3 국방 분야 LLM 활용 연구 동향
   2.4 기존 연구의 한계 및 본 연구의 차별성

3. 시스템 설계
   3.1 전체 아키텍처
   3.2 전투 시뮬레이션 엔진
       3.2.1 헥사곤 격자 전장 모델 및 전장 안개(Fog of War)
       3.2.2 선형 Lanchester 법칙 기반 교전 판정 (1-step 이산 근사)
   3.3 LLM 에이전트 설계
       3.3.1 FM 3-90 기반 MDMP 단일 CoT 프롬프트 구조
       3.3.2 역할별 시스템 프롬프트 (Blue / Red / White Cell)
       3.3.3 구조화된 출력(JSON) 및 강건한 행동 파싱
   3.4 M&S-에이전트 연동 인터페이스

4. 실험 설계
   4.1 실험 환경 (하드웨어, 모델, 양자화, mlx-lm API)
   4.2 시나리오 설계 및 Phase 1 검증 결과 (5종)
   4.3 독립변수, 종속변수, 통제변수
   4.4 평가 지표 정의 (DCR, Action Entropy, ESI, TRS 등)
   4.5 베이스라인 및 비교 조건
   4.6 합성 데이터 생성 및 검증

5. 실험 결과 및 분석
   5.1 교리 준수율 분석 (RQ1)
   5.2 모델 간 행동 프로파일 비교 (RQ2)
   5.3 전술 다양성 비교 실험 (RQ3)
   5.4 전투 결과 및 성능 분석
   5.5 정성적 사례 분석 (S2·S4 Rule vs LLM 행동 차이 포함)

6. 토의
   6.1 주요 발견 요약
   6.2 소형 LLM의 전술적 추론 능력과 한계
   6.3 국방 M&S에의 시사점
   6.4 연구 제한점 (선형법칙 근사, 7B 모델 한계, 합성 시나리오)
   6.5 향후 연구 방향

7. 결론

참고문헌

부록
   A. 시스템 프롬프트 전문 (Blue / Red / White Cell)
   B. 시나리오 상세 파라미터
   C. 교리 준수 평가 루브릭 (6개 원칙 체크리스트)
   D. 추가 실험 결과 테이블
```

### 6.3 투고 추천 저널 (KCI 기준)

| 순위 | 저널명 | 발행 기관 | 적합 근거 |
|------|--------|-----------|-----------|
| **1순위** | **한국시뮬레이션학회 논문지** | 한국시뮬레이션학회 | M&S + AI 융합 연구에 가장 직접적으로 부합. Lanchester 모형, 워게임 시뮬레이션 관련 논문 게재 실적 풍부 |
| **2순위** | **한국산학기술학회논문지 (JKAIS)** | 한국산학기술학회 | LLM + 국방 + M&S 융합 주제 수용 가능, 빠른 심사 |
| **3순위** | **국방과 보안** | 국방보안연구소 | 국방 LLM 특화 저널, 관련 논문 다수 |

---

## 부록: 확정 코드 스켈레톤

### A. Lanchester Combat Resolver (실제 구현)

```python
# src/wargame/combat/lanchester.py (핵심 로직)
# 선형 Lanchester 법칙 1-step 이산 근사 (scipy.odeint 미사용)

def resolve_lanchester(blue_strength, red_strength, *, config=None,
                       blue_defense_modifier=1.0, red_defense_modifier=1.0,
                       stochastic=None, seed=None, rng=None):
    config = config or LanchesterConfig()  # alpha=beta=0.05, noise_std=0.1
    stochastic = config.stochastic if stochastic is None else stochastic

    base_blue_loss = (
        config.red_attrition_rate * red_strength * config.time_step / blue_defense_modifier
    )
    base_red_loss = (
        config.blue_attrition_rate * blue_strength * config.time_step / red_defense_modifier
    )

    blue_noise = red_noise = 0.0
    if stochastic:
        local_rng = Random(seed) if rng is None else rng
        blue_noise = local_rng.gauss(0.0, config.noise_std * base_blue_loss)
        red_noise  = local_rng.gauss(0.0, config.noise_std * base_red_loss)

    blue_loss = _clamp_loss(base_blue_loss + blue_noise, blue_strength)
    red_loss  = _clamp_loss(base_red_loss  + red_noise,  red_strength)

    return LanchesterOutcome(
        blue_start=blue_strength, red_start=red_strength,
        blue_loss=blue_loss, red_loss=red_loss,
        blue_remaining=blue_strength - blue_loss,
        red_remaining=red_strength  - red_loss,
        ...
    )
```

### B. LLM Agent 추론 래퍼 — MLX (실제 구현)

```python
# src/wargame/agents/backends/mlx_backend.py
# mlx-lm 최신 API: make_sampler + sampler= 인자

from mlx_lm.sample_utils import make_sampler

sampler = make_sampler(temp=config.temperature)   # temperature 제어
output = mlx_lm.generate(
    self._model,
    self._tokenizer,
    prompt=prompt,
    max_tokens=config.max_tokens,
    sampler=sampler,   # temp= 직접 전달 방식 폐기
    verbose=False,
)
```

### C. JSONL 로그 구조 (실제 키 명세)

```jsonl
// 컨텍스트 레코드 (첫 줄)
{"context": {"run_id": "...", "scenario_id": "s1_open_encounter",
             "initial_force_totals": {"blue": 300, "red": 300}}}

// 턴 레코드
{"turn": 4, "terminal": false,
 "state": {"units": {"blue-a": {"faction": "blue", "strength": 292,
                                "position": {"q": 5, "r": 3}}, ...},
           "terrain_by_hex": {"5,3": "open", ...}},
 "combat": {"casualties_by_unit": {"blue-a": 8, "red-a": 3},
            "summary": "...", "winner": null},
 "metadata": {
   "blue": {"reasoning": "ASSESS: ... DEVELOP: COA A ... DECIDE: ...",
            "doctrine_reference": "FM 3-90, Ch.3",
            "used_fallback": false},
   "white_cell": {"metadata": {"scores": {"doctrine_compliance": 0.833,
                                           "tactical_rationality": 4.2}}}
 }}
```

> **주의**: `combat.blue_loss`, `combat.red_loss` 키는 존재하지 않음.
> 반드시 `combat.casualties_by_unit` 딕셔너리를 사용해야 함.
