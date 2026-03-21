# 상세 연구 설계서

## LLM 기반 다중 에이전트 워게임 시뮬레이션 프레임워크: 전술적 의사결정 행동 분석을 위한 접근

> **연구 유형**: 시스템 구현 + 실험 연구
> **대상 저널**: 국내 KCI 등재 저널
> **컴퓨팅 환경**: MacBook Pro M4 16GB / Google Colab Pro (로컬 추론 전용, API 미사용)

---

## 1. 연구 문제 정의

### 1.1 연구 배경

전투 시뮬레이션과 워게임은 군사 교육·훈련 및 전투실험의 핵심 도구이다. 그러나 기존 컴퓨터 기반 워게임은 (1) 적/아군의 행동이 사전 정의된 스크립트나 규칙에 의존하여 예측 가능하며, (2) 정성적(qualitative) 워게임의 자동화가 불가능하여 대규모 반복 실험에 한계가 있고, (3) 의사결정의 추론 과정이 블랙박스로 남아 교육적 분석이 어렵다. 최근 대규모 언어모델(LLM)의 발전은 자연어 기반의 전술 추론과 상황 적응적 의사결정을 가능하게 하여, 이러한 한계를 극복할 잠재력을 보여주고 있다.

### 1.2 연구 질문 (Research Questions)

**RQ1. LLM 기반 다중 에이전트는 턴제 전투 시뮬레이션 환경에서 군사 교리에 부합하는 전술적 의사결정을 수행할 수 있는가?**
- 교리 준수율(Doctrine Compliance Rate)과 전술적 합리성(Tactical Rationality)을 정량 측정
- LLM 에이전트의 의사결정이 무작위가 아닌 상황 맥락에 기반함을 검증

**RQ2. 서로 다른 소형 LLM(7B급) 간 전술적 의사결정 특성에는 어떤 차이가 있으며, 모델 선택이 워게임 시뮬레이션 결과에 미치는 영향은 무엇인가?**
- Qwen2.5-7B, Mistral-7B, Llama-3.1-8B 등 3종 이상 모델 비교
- 공격 성향(Escalation Tendency), 방어 선호도, 전술 다양성 등 행동 프로파일 분석

**RQ3. LLM 에이전트 기반 워게임 시뮬레이션은 기존 스크립트/규칙 기반 워게임 대비 전술적 다양성과 예측 불가능성 측면에서 유의미한 차이를 보이는가?**
- 동일 시나리오 100회 반복 시 행동 분포의 엔트로피 비교
- 스크립트 기반 / 규칙 기반 / LLM 기반의 3-Way 비교 실험

### 1.3 기존 연구와의 차별성 (Gap 분석)

| 차원 | 기존 연구 | 본 연구의 차별점 |
|------|-----------|------------------|
| **시뮬레이션 연동** | WarAgent(Hua et al., 2023): 국가 단위 전략 수준 시뮬레이션, 전투 결과는 LLM이 자체 판정 | 전술 수준(중대~대대)의 전투 시뮬레이션 엔진을 별도 구축하여 Lanchester 모형 기반 정량적 교전 결과 산출 |
| **모델 규모** | 대부분 GPT-4, Claude 등 대형 상용 모델 의존 | 7B급 양자화 로컬 모델만 사용, 제한된 컴퓨팅 환경에서의 실현 가능성 검증 |
| **평가 방법** | 역사적 사건 재현 정확도 중심 | 교리 준수율, 전술 다양성 엔트로피, 행동 프로파일 등 다면적 정량 평가 |
| **에이전트 구조** | 단일 프롬프트 기반 의사결정 | 상황인식→방책수립→결심의 MDMP(Military Decision-Making Process) 기반 다단계 추론 |
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
│                  │  │ Map/    │  │ Combat   │  │ State     │  │   │
│                  │  │ Terrain │  │ Resolver │  │ Manager   │  │   │
│                  │  │ Module  │  │(Lanch.)  │  │           │  │   │
│                  │  └─────────┘  └──────────┘  └───────────┘  │   │
│                  └──────────┬───────────────────────┬──────────┘   │
│                             │ State                 │ Actions      │
│                             ▼                       ▲              │
│                  ┌──────────────────────────────────────────────┐   │
│                  │         Agent Orchestrator                   │   │
│                  │                                              │   │
│                  │  ┌──────────────┐  ┌──────────────────────┐ │   │
│                  │  │ State-to-Text│  │ Action Parser        │ │   │
│                  │  │ Converter    │  │ (JSON → Engine Cmd)  │ │   │
│                  │  └──────┬───────┘  └──────────▲───────────┘ │   │
│                  │         │ Text                 │ JSON        │   │
│                  │         ▼                      │             │   │
│                  │  ┌────────────────────────────────────────┐  │   │
│                  │  │        LLM Agent Pool                  │  │   │
│                  │  │                                        │  │   │
│                  │  │  ┌──────────┐ ┌──────────┐ ┌────────┐ │  │   │
│                  │  │  │Blue Force│ │Red Force │ │White   │ │  │   │
│                  │  │  │ Agent    │ │ Agent    │ │Cell    │ │  │   │
│                  │  │  │(MDMP)   │ │(MDMP)    │ │Agent   │ │  │   │
│                  │  │  └──────────┘ └──────────┘ └────────┘ │  │   │
│                  │  └────────────────────────────────────────┘  │   │
│                  └─────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     Logger & Analyzer                        │   │
│  │  - Turn-by-turn decision log (JSON)                         │   │
│  │  - CoT reasoning trace                                       │   │
│  │  - Combat outcome statistics                                 │   │
│  │  - Doctrine compliance scoring                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 모듈별 상세 설계

#### 2.2.1 Simulation Engine

**역할**: 전장 상태 관리, 교전 결과 판정, 턴 진행 통제

**핵심 구성 요소**:

**(a) Map/Terrain Module**
- 헥사곤 격자(Hexagonal Grid) 기반 전장 지도 표현
- 지형 유형: 평지(Open), 산악(Mountain), 시가지(Urban), 삼림(Forest), 하천(River)
- 각 지형에 방어 보너스 계수(Defense Modifier) 및 이동 비용(Movement Cost) 부여
- 맵 크기: 20×15 헥스 (전술 수준 소규모 교전에 적합)

**(b) Combat Resolver (Lanchester 모형 기반)**
- 기본 모형: 확장 Lanchester 제곱법칙 (Stochastic Lanchester Square Law)
- 전투력 산출식:

```
dB/dt = -α · R(t)    (Blue 손실률 = Red 전투효율 × Red 잔존 병력)
dR/dt = -β · B(t)    (Red 손실률 = Blue 전투효율 × Blue 잔존 병력)
```

- 지형 보정: α' = α × terrain_modifier_blue, β' = β × terrain_modifier_red
- 확률적 요소: 각 턴 교전 결과에 정규분포 노이즈 추가 (μ=0, σ=0.1×base_attrition)
- 교전 종료 조건: 일방 병력 30% 이하 → 전투 종료 판정

**(c) State Manager**
- 턴별 전장 상태를 Python 딕셔너리/dataclass로 관리
- 상태 정보: 각 부대의 {위치(hex), 병력수, 전투력지수, 보급상태, 사기, 전투태세}
- 전장 안개(Fog of War): 각 에이전트에게 가시 범위 내 적 정보만 제공

**인터페이스 (Python 클래스)**:

```python
class SimulationEngine:
    def get_state(self, faction: str) -> dict       # 해당 진영 관점의 전장 상태 반환
    def execute_actions(self, actions: list) -> dict  # 행동 실행 후 결과 반환
    def resolve_combat(self) -> dict                  # 교전 판정 및 손실 계산
    def advance_turn(self) -> dict                    # 턴 진행, 보급/사기 업데이트
    def is_terminal(self) -> bool                     # 종료 조건 확인
    def get_log(self) -> list                         # 전체 턴 로그 반환
```

#### 2.2.2 Agent Orchestrator

**역할**: LLM 에이전트와 시뮬레이션 엔진 간 중재, 턴 순서 관리

**핵심 구성 요소**:

**(a) State-to-Text Converter**
- 시뮬레이션 엔진의 Python 상태 객체를 자연어 상황 보고서(Situation Report)로 변환
- 변환 템플릿 예시:

```
## 현재 상황 보고 (Turn {n})
### 아군(Blue Force) 현황
- 1중대: 위치 H7, 병력 120명(전투력 85%), 전투태세: 공격준비
- 2중대: 위치 G9, 병력 95명(전투력 72%), 전투태세: 방어
### 적군(Red Force) 관측 정보
- 적 1개 중대 규모: 위치 E5 부근 관측, 추정 병력 100±20명
- 적 화력 지원 징후: D3 지역에서 포격 관측
### 지형 정보
- H7→F5 경로: 산악 지형, 이동 비용 2배, 방어 보너스 +30%
- G9→F7 경로: 삼림 지형, 이동 비용 1.5배, 은엄폐 양호
### 임무
- 목표: Turn 15까지 E5 지역 확보
- 제한사항: H열 이남으로 철수 불가
```

**(b) Action Parser**
- LLM 출력(JSON)을 시뮬레이션 엔진 명령으로 변환
- Structured Output을 위한 JSON Schema 사전 정의:

```json
{
  "reasoning": "적 방어진지 정면 공격은 손실이 크므로 우회기동 선택",
  "actions": [
    {
      "unit": "1st_company",
      "action_type": "move",
      "target_hex": "F6",
      "posture": "attack"
    },
    {
      "unit": "2nd_company",
      "action_type": "support_by_fire",
      "target_hex": "E5",
      "posture": "defense"
    }
  ],
  "doctrine_reference": "FM 3-90: 우회기동은 적 방어의 측면 또는 후방을 공격"
}
```

- 파싱 실패 시 최대 2회 재시도 후 기본 행동(현 위치 방어) 실행
- 잘못된 좌표, 불가능한 행동 등에 대한 Validation 로직 포함

#### 2.2.3 LLM Agent Pool

**공통 에이전트 구조 (MDMP 기반 3단계 추론)**:

```
[Stage 1: 상황판단 (Situation Assessment)]
  입력: State-to-Text 변환 결과
  출력: 적 위협 분석, 아군 전투력 평가, 핵심 지형 식별

[Stage 2: 방책수립 (COA Development)]
  입력: Stage 1 출력 + 교리 가이드라인(System Prompt)
  출력: 2~3개의 행동 방책(COA) 후보

[Stage 3: 결심 (Decision)]
  입력: Stage 2의 COA 후보들 + 평가 기준
  출력: 최종 선택 COA + 실행 명령(JSON)
```

**주의**: 3단계를 단일 프롬프트의 CoT로 구현 (3회 별도 호출 X → 1회 호출 내 섹션 구분)
→ 추론 시간 및 메모리 절약

**역할별 시스템 프롬프트 설계**:

**(a) Blue Force Agent (아군 지휘관)**

```
You are a Blue Force battalion commander in a tactical wargame simulation.
Your mission is to achieve the assigned objective while minimizing friendly casualties.

DOCTRINE GUIDELINES (Based on FM 3-90 Tactics):
- Concentration: Mass combat power at the decisive point
- Surprise: Avoid predictable patterns of operation
- Security: Never leave flanks exposed without observation
- Maneuver: Use terrain to gain positional advantage

DECISION PROCESS:
1. ASSESS: Analyze enemy disposition, terrain, and friendly status
2. DEVELOP: Generate 2-3 possible courses of action
3. DECIDE: Select the best COA and output as structured JSON

You must respond ONLY with valid JSON matching the provided schema.
Include your reasoning in the "reasoning" field.
```

**(b) Red Force Agent (적군 지휘관)**

```
You are a Red Force commander defending against a Blue Force attack.
Your tactics are based on defensive doctrine principles.

DOCTRINE GUIDELINES:
- Defense in depth: Echelon forces to absorb and attrite attackers
- Counterattack: Strike when the enemy is overextended
- Terrain utilization: Maximize defensive advantage of key terrain
- Deception: Mislead the enemy about main defensive positions

[Same decision process structure as Blue Force]
```

**(c) White Cell Agent (심판/판정관)**

```
You are the White Cell adjudicator overseeing this wargame.
Your role is to evaluate each turn's actions and provide:
1. Assessment of tactical soundness (1-5 scale)
2. Doctrine compliance check (pass/fail per action)
3. Narrative summary of the turn's events
4. Any rule violations or unrealistic actions

Evaluate based on military tactical principles, not on outcome.
A well-reasoned action that fails is better than a lucky poor decision.
```

### 2.3 M&S 환경 연동 방식

```
[Turn Loop Sequence]

1. Engine.get_state("blue") → State-to-Text → Blue Agent prompt
2. Blue Agent inference (local LLM) → JSON actions
3. Action Parser validates → Engine.execute_actions(blue_actions)
4. Engine.get_state("red") → State-to-Text → Red Agent prompt
5. Red Agent inference (local LLM) → JSON actions
6. Action Parser validates → Engine.execute_actions(red_actions)
7. Engine.resolve_combat() → Lanchester 모형 교전 판정
8. Engine.advance_turn() → 보급/사기 업데이트
9. Logger records: {turn, states, actions, reasoning, outcomes}
10. White Cell Agent evaluates turn → doctrine_score, narrative
11. Engine.is_terminal() → True이면 종료, False이면 1로 복귀
```

**핵심 설계 원칙**:
- **LLM은 의사결정만 수행**: 교전 결과는 반드시 Lanchester 모형이 판정 (LLM의 환각 방지)
- **전장 안개(Fog of War) 적용**: 에이전트는 관측 가능한 범위 내 정보만 수신
- **추론 과정 완전 기록**: 모든 CoT 추론 과정을 JSON 로그에 보존하여 사후 분석 가능

---

## 3. 구현 계획

### 3.1 기술 스택

| 구분 | 기술 | 선정 근거 |
|------|------|-----------|
| **LLM 추론 (Mac)** | MLX + mlx-lm | Apple Silicon 최적화, M4에서 Q4 7B 모델 60~70 tok/s |
| **LLM 추론 (Colab)** | vLLM 또는 Transformers + BitsAndBytes | A100/V100 활용, 4-bit 양자화 |
| **LLM 모델** | Qwen2.5-7B-Instruct-MLX-4bit (Mac), GGUF Q4_K_M (Colab llama.cpp) | JSON 출력 능력 우수, 다국어 지원 |
| **비교 모델** | Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct | 아키텍처 다양성 확보 |
| **시뮬레이션 엔진** | Python 3.11+, NumPy, SciPy | 순수 Python으로 경량 구현 |
| **지도 시스템** | 자체 구현 (HexGrid 클래스) | 의존성 최소화 |
| **Lanchester 모형** | SciPy.integrate.odeint | 미분방정식 수치 해법 |
| **데이터 관리** | JSON Lines (.jsonl) | 턴별 로그 저장, 분석 용이 |
| **시각화** | Matplotlib, Seaborn | 전장 상황도 및 실험 결과 차트 |
| **통계 분석** | SciPy.stats, pandas | 가설 검정 및 기술 통계 |
| **에이전트 프레임워크** | 자체 구현 (LangChain 미사용) | 경량화, 커스터마이징 자유도 |

### 3.2 단계별 구현 순서

#### Phase 1: 시뮬레이션 엔진 구축 (Week 1-2) — MacBook M4

```
[Week 1]
├── HexGrid 클래스 구현 (좌표 체계, 인접 탐색, 거리 계산)
├── Terrain 모듈 구현 (지형 유형별 수정치 테이블)
├── Unit/Force 데이터 모델 정의 (dataclass)
└── State Manager 기본 구현

[Week 2]
├── Lanchester Combat Resolver 구현 및 단위 테스트
│   ├── 결정론적(deterministic) 버전 먼저 구현
│   └── 확률적(stochastic) 버전 추가
├── 턴 진행 로직 구현
├── Fog of War 로직 구현
└── 엔진 통합 테스트 (스크립트 에이전트로 10회 실행)
```

#### Phase 2: LLM 에이전트 시스템 구축 (Week 3-4) — MacBook M4 + Colab

```
[Week 3]
├── MLX 환경 구축 (Mac)
│   ├── mlx-lm 설치, Qwen2.5-7B-Instruct-4bit 다운로드
│   └── 추론 속도 및 메모리 사용량 벤치마크
├── State-to-Text Converter 구현
├── Action Parser + JSON Schema Validator 구현
└── Blue/Red/White Agent 시스템 프롬프트 초안 작성

[Week 4]
├── Agent Orchestrator 구현 (턴 루프 통합)
├── 프롬프트 반복 개선 (10회 수동 테스트)
│   ├── JSON 출력 안정성 확보
│   ├── 교리 참조 품질 향상
│   └── 불가능한 행동 방지 가드레일 추가
├── Colab 환경 구성 (Mistral-7B, Llama-3.1-8B 추가)
└── Logger 모듈 구현 (JSONL 형식)
```

#### Phase 3: 베이스라인 시스템 구현 (Week 5) — MacBook M4

```
[Week 5]
├── Script Agent 구현 (사전 정의 행동 시퀀스)
│   └── 3종 스크립트: 정면공격, 우회기동, 지연전
├── Rule-Based Agent 구현 (if-then 규칙 엔진)
│   ├── 전투력 비율 기반 공격/방어 판단
│   ├── 거리 기반 이동 우선순위
│   └── 손실률 기반 철수 판단
└── 베이스라인 에이전트 통합 테스트 (각 50회 실행)
```

#### Phase 4: 대규모 실험 실행 (Week 6-8) — Colab Pro (주) + MacBook (부)

```
[Week 6-7]
├── 실험 시나리오 5종 × 3 LLM 모델 × 3 에이전트 유형 = 45개 조건
├── 조건당 100회 반복 → 총 4,500회 시뮬레이션 실행
│   ├── Colab A100: LLM 에이전트 실험 (세션당 ~300회)
│   └── MacBook M4: 스크립트/규칙 에이전트 병렬 실행
├── 실행 중간 점검: 로그 무결성, 파싱 오류율 모니터링
└── White Cell 평가 실행 (별도 배치)

[Week 8]
├── 결과 데이터 취합 및 정제
├── 통계 분석 실행
└── 시각화 생성
```

#### Phase 5: 논문 작성 (Week 9-12)

```
[Week 9-10]
├── 실험 결과 해석 및 Discussion 초안
├── Related Work 조사 및 정리
└── 시스템 아키텍처 다이어그램 작성

[Week 11-12]
├── 전체 논문 초안 완성
├── 자체 검토 및 수정
└── KCI 저널 투고
```

### 3.3 예상 병목 지점 및 대안

| 병목 지점 | 문제 상황 | 대안 |
|-----------|-----------|------|
| **JSON 출력 실패** | 7B 모델이 유효하지 않은 JSON 생성 | (1) JSON repair 라이브러리(json-repair) 적용, (2) Regex 기반 후처리, (3) 재시도 최대 2회 |
| **MacBook 메모리 부족** | 7B Q4 모델(~4GB) + Python 엔진 + 시스템 메모리 경합 | (1) MLX의 lazy evaluation 활용, (2) 시뮬레이션 배치 사이 모델 언로드, (3) 컨텍스트 길이 2048로 제한 |
| **Colab 세션 단절** | 장시간 실험 중 세션 타임아웃 | (1) 실험을 100회 단위 배치로 분할, (2) 매 배치 완료 시 Google Drive 자동 저장, (3) 체크포인트에서 이어서 실행 |
| **추론 속도** | 매 턴 2~3회 LLM 호출 × 15턴 × 100회 반복 | (1) 3단계 MDMP를 단일 프롬프트 CoT로 통합 (호출 횟수 절감), (2) White Cell 평가는 게임 종료 후 배치 처리, (3) Mac MLX에서 Q4 7B는 60+ tok/s로 턴당 ~5초 예상 |
| **모델 간 프롬프트 호환성** | 모델마다 프롬프트 포맷(ChatML, Llama 포맷 등)이 다름 | 모델별 chat template을 자동 적용하는 래퍼 함수 구현, 핵심 시스템 프롬프트는 동일 내용 유지 |
| **교리 환각(Hallucination)** | 존재하지 않는 교리 인용 | (1) 시스템 프롬프트에 실제 교리 조항을 직접 포함(RAG 불필요), (2) White Cell 에이전트가 교리 인용 정확성 검증 |

### 3.4 컴퓨팅 리소스 예산

```
[MacBook Pro M4 16GB 작업량]
- 시뮬레이션 엔진 개발 및 테스트: 전 과정
- MLX 기반 Qwen2.5-7B 추론: 주 실험 모델
  - 예상 속도: Q4_K_M → ~65 tok/s (generation)
  - 턴당 출력 ~300 tokens → ~5초/턴
  - 15턴 게임 1회: ~75초 (Blue+Red 각 5초 × 15턴)
  - 100회 반복: ~2시간
- 스크립트/규칙 베이스라인: 100회 × 5시나리오 → 수 분

[Google Colab Pro A100 작업량]
- Mistral-7B, Llama-3.1-8B 추론 실험
  - vLLM + 4-bit 양자화 → ~80+ tok/s 예상
  - 100회 × 5시나리오 × 2모델 = 1,000회
  - 예상 소요: 세션당 4~5시간 × 4~5세션
- White Cell 배치 평가: 게임 종료 후 전체 로그 일괄 처리
```

---

## 4. 실험 설계

### 4.1 변수 정의

#### 독립변수 (Independent Variables)

| 변수 | 수준 | 설명 |
|------|------|------|
| **에이전트 유형** | 3수준: Script / Rule-Based / LLM | 의사결정 메커니즘의 종류 |
| **LLM 모델** | 3수준: Qwen2.5-7B / Mistral-7B / Llama-3.1-8B | 모델 아키텍처 및 학습 데이터 차이 |
| **시나리오** | 5수준: 아래 5.1절 참조 | 전술 상황의 다양성 |

#### 종속변수 (Dependent Variables)

| 변수 | 측정 방법 | 단위 |
|------|-----------|------|
| **교리 준수율 (DCR)** | White Cell 에이전트의 턴별 교리 준수 판정 비율 | % (0~100) |
| **전술적 합리성 점수 (TRS)** | White Cell 에이전트의 5점 척도 평균 | 1.0~5.0 |
| **행동 다양성 (Action Entropy)** | 100회 반복 시 동일 상황에서의 행동 분포 Shannon Entropy | bits |
| **전투 결과** | Blue Force 승률, 평균 잔존 병력 비율 | %, % |
| **공격 성향 지수 (ESI)** | 공격 행동 비율 / 전체 행동 수 | 0.0~1.0 |
| **전술 전환 빈도 (TTF)** | 게임 내 전술 변경 횟수 (공격↔방어↔이동) | 회/게임 |
| **JSON 파싱 성공률** | 유효한 JSON 출력 비율 | % |
| **추론 시간** | 턴당 LLM 추론 소요 시간 | 초 |

#### 통제변수 (Control Variables)

| 변수 | 통제 방법 |
|------|-----------|
| 초기 병력 배치 | 시나리오별 고정 (동일 초기 조건) |
| 지형 맵 | 시나리오별 고정 맵 |
| Lanchester 전투 효율 계수 | α=0.05, β=0.05 고정 (대칭 전투력) |
| 확률적 노이즈 시드 | 100회 반복 시 동일 시드 세트 사용 |
| 최대 턴 수 | 20턴으로 고정 |
| 컨텍스트 윈도우 | 2048 토큰으로 고정 |
| Temperature | 0.7 고정 (모든 LLM 모델 동일) |

### 4.2 평가 지표(Metrics) 상세

**(1) 교리 준수율 (Doctrine Compliance Rate, DCR)**
- 정의: 전체 턴에서 교리 원칙에 부합하는 의사결정의 비율
- 평가 기준 (6개 교리 원칙):
  - 집중(Concentration): 주공 방향에 전투력 2/3 이상 집중 여부
  - 경계(Security): 노출된 측면에 관측/경계 배치 여부
  - 기동(Maneuver): 적 강점 회피, 약점 지향 여부
  - 간명(Simplicity): 1개 턴에 3개 이하의 동시 기동 여부
  - 목표(Objective): 임무 목표 방향으로의 일관된 진전 여부
  - 통합(Unity of Command): 부대 간 상호 지원 가능 거리 유지 여부
- 산출: (준수 판정 수) / (전체 판정 수) × 100

**(2) 행동 다양성 (Action Entropy)**
- 정의: 동일 초기 상황에서 100회 반복 시 첫 턴 행동의 Shannon Entropy
- 산출: H = -Σ p(a) · log₂(p(a)), where a ∈ {가능한 행동 집합}
- 의미: 높을수록 비예측적, 낮을수록 결정론적
- Script Agent: H ≈ 0 (완전 결정론적), Rule-Based: H < 1, LLM: H > 1 예상

**(3) 공격 성향 지수 (Escalation Sensitivity Index, ESI)**
- 정의: 전체 행동 중 공격적 행동(이동+공격, 포격, 돌격)의 비율
- 산출: ESI = N_offensive / N_total
- 선행연구 근거: LLM Wargaming(Rivera et al., 2024)에서 LLM의 폭력 성향 편향 발견

### 4.3 베이스라인 모델 선정 기준

| 베이스라인 | 구현 방식 | 선정 근거 |
|-----------|-----------|-----------|
| **Script Agent** | 사전 정의된 행동 시퀀스 실행 (시나리오별 3종 스크립트) | 최소 기준선: 무작위보다 나은 고정 전략의 성능 |
| **Rule-Based Agent** | IF-THEN 규칙 엔진 (전투력 비율, 거리, 손실률 기반 15개 규칙) | 기존 M&S에서 가장 널리 사용되는 AI 적용 방식 |
| **Random Agent** | 유효한 행동 중 균일 무작위 선택 | 하한 기준선: LLM이 무작위보다 유의미하게 나은지 검증 |

### 4.4 통계 검정 계획

- RQ1: One-sample t-test (DCR > 50%, 즉 무작위 이상) + Cohen's d 효과 크기
- RQ2: One-way ANOVA (3 LLM 모델 간 TRS, ESI 차이) + Tukey HSD 사후 검정
- RQ3: Kruskal-Wallis test (Script vs Rule vs LLM의 Action Entropy) + Dunn 사후 검정
- 유의수준: α = 0.05, 다중 비교 보정: Bonferroni

---

## 5. 합성 데이터 생성 전략

### 5.1 시나리오 설계 (5종)

| ID | 시나리오명 | 상황 설정 | 핵심 전술 요소 |
|----|-----------|-----------|---------------|
| S1 | **평지 조우전** | 양측 각 3개 중대, 평지, 조우 상황 | 기동의 자유, 측면 기동 가능 |
| S2 | **산악 방어진지 공격** | Blue 공격 / Red 방어, 산악 지형 | 지형 활용, 공격 경로 선택 |
| S3 | **시가지 전투** | 양측 진입, 시가지 중심 확보 | 근접 전투, 건물 활용 |
| S4 | **하천 도하 작전** | Blue 도하 / Red 도하 저지 | 취약 시점의 전투력 집중 |
| S5 | **포위 돌파** | Red 포위 / Blue 돌파 시도 | 비대칭 상황, 위기 의사결정 |

### 5.2 시나리오 생성 절차

```
Step 1: 시나리오 파라미터 정의
  ├── 맵 생성: 20×15 헥스 그리드, 지형 배치 (수동 설계 5종)
  ├── 초기 병력 배치: 시나리오별 고정
  ├── 임무 목표: 시나리오별 텍스트 + 목표 헥스 좌표
  └── 종료 조건: 목표 점령 또는 20턴 경과 또는 일방 전멸

Step 2: 파라미터 변이(Perturbation)를 통한 확장
  ├── 병력 비율 변이: 1:1, 1.5:1, 2:1 (공격:방어)
  ├── 전투 효율 비대칭: α ≠ β 조건 추가 (선택적)
  └── 초기 배치 미세 조정: ±1 헥스 랜덤 이동 (100회 반복 시)

Step 3: 시나리오 검증
  ├── Rule-Based Agent로 10회 사전 실행
  ├── 교착(stalemate) 빈도 확인 (<20%이면 통과)
  └── 평균 게임 길이 확인 (8~18턴이면 적정)
```

### 5.3 데이터 품질 검증 방법

| 검증 항목 | 방법 | 기준 |
|-----------|------|------|
| **시나리오 균형성** | Rule-Based Agent 대칭 실행 50회 → 승률 분포 | 40~60% 범위 내 |
| **LLM 출력 유효성** | JSON Schema 자동 검증 | 파싱 성공률 ≥ 90% |
| **교리 평가 일관성** | 동일 로그에 대한 White Cell 3회 반복 평가 → 평가자내 일치도 | Krippendorff's α ≥ 0.7 |
| **시뮬레이션 재현성** | 동일 시드, 동일 행동 → 동일 결과 확인 | 100% 일치 |
| **전문가 표본 검증** | 전체 로그 중 10% 무작위 추출 → 군사 전문가 2인 평가 | 전문가-White Cell 일치도 ≥ κ=0.6 |

### 5.4 기존 오픈 데이터셋 활용 방법

| 데이터셋 | 활용 방법 |
|----------|-----------|
| **tactical-military-reasoning (HuggingFace)** | 시스템 프롬프트의 Few-shot 예시로 활용 (전술 추론 스타일 참조) |
| **FM 3-90, FM 3-21.8 (미군 공개 교범)** | 교리 원칙 추출 → 시스템 프롬프트 및 DCR 평가 기준 구축 |
| **WarAgent 시나리오 데이터 (GitHub)** | 시나리오 설계 참조 (전략 수준 → 전술 수준으로 스케일 다운) |
| **Panopticon AI 플랫폼 (GitHub)** | 시뮬레이션 엔진 설계 참조 (Gymnasium 호환 인터페이스 참고) |

---

## 6. 예상 결과 및 KCI 논문 구성

### 6.1 예상 핵심 기여 결과

**(1) LLM 에이전트의 전술적 의사결정 능력 입증**
- 7B급 소형 모델도 교리 준수율 60~75% 수준에서 전술적으로 유의미한 의사결정 가능 예상
- Random Agent(~30%) 및 Script Agent(고정값) 대비 통계적으로 유의미한 차이

**(2) 모델 간 행동 프로파일 차이 규명**
- Qwen2.5: 상대적으로 보수적/방어적 성향 예상 (중국어 학습 데이터의 방어 교리 영향)
- Mistral: 공격적 성향이 강할 가능성 (서양 군사 문화 반영)
- Llama-3.1: 중간 성향, 가장 균형잡힌 행동 분포 예상

**(3) LLM 기반 워게임의 실현 가능성 검증**
- 로컬 7B 모델만으로도 완전 자동화된 워게임 시뮬레이션 반복 실행 가능
- 국방 폐쇄망 환경에서의 적용 가능성 시사

**(4) 전술 다양성 측면의 우위**
- LLM Agent의 Action Entropy가 Rule-Based 대비 2~3배 높을 것으로 예상
- 동일 상황에서도 다양한 전술적 대안을 생성하여 워게임의 교육적 가치 향상

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
       3.2.1 헥사곤 격자 전장 모델
       3.2.2 Lanchester 기반 교전 판정 모형
   3.3 LLM 에이전트 설계
       3.3.1 MDMP 기반 다단계 추론 구조
       3.3.2 역할별 시스템 프롬프트 설계
       3.3.3 구조화된 출력(JSON) 및 행동 파싱
   3.4 M&S-에이전트 연동 인터페이스

4. 실험 설계
   4.1 실험 환경 (하드웨어, 모델, 양자화)
   4.2 시나리오 설계 (5종)
   4.3 독립변수, 종속변수, 통제변수
   4.4 평가 지표 정의
   4.5 베이스라인 및 비교 조건
   4.6 합성 데이터 생성 및 검증

5. 실험 결과 및 분석
   5.1 교리 준수율 분석 (RQ1)
   5.2 모델 간 행동 프로파일 비교 (RQ2)
   5.3 전술 다양성 비교 실험 (RQ3)
   5.4 전투 결과 및 성능 분석
   5.5 정성적 사례 분석 (대표 게임 2~3건 상세 분석)

6. 토의
   6.1 주요 발견 요약
   6.2 소형 LLM의 전술적 추론 능력과 한계
   6.3 국방 M&S에의 시사점
   6.4 연구 제한점
   6.5 향후 연구 방향

7. 결론

참고문헌

부록
   A. 시스템 프롬프트 전문
   B. 시나리오 상세 파라미터
   C. 교리 준수 평가 루브릭
   D. 추가 실험 결과 테이블
```

### 6.3 투고 추천 저널 (KCI 기준)

| 순위 | 저널명 | 발행 기관 | 적합 근거 |
|------|--------|-----------|-----------|
| **1순위** | **한국시뮬레이션학회 논문지 (Journal of the Korea Society for Simulation)** | 한국시뮬레이션학회 | M&S + AI 융합 연구에 가장 직접적으로 부합. Lanchester 모형, 워게임 시뮬레이션 관련 논문 게재 실적 풍부. 시스템 구현 + 실험 형태의 논문을 선호 |
| **2순위** | **한국산학기술학회논문지 (Journal of KAIS)** | 한국산학기술학회 | 2025년 국방 AI 동향 분석 등 관련 논문 게재 실적. 분야 제한이 넓어 LLM + 국방 + M&S 융합 주제 수용 가능. 비교적 빠른 심사 기간 |
| **3순위** | **국방과 보안 (Defense and Security)** | 국방보안연구소 | 국방 LLM 특화 저널. RAG-LLM 기반 훈련 계획 생성, LLM 활용 문제점 등 직접 관련 논문 다수 게재. 2025년에 LLM+국방 특집 논문 활발 |

---

## 부록: 핵심 코드 스켈레톤

### A. Lanchester Combat Resolver (핵심 로직)

```python
import numpy as np
from scipy.integrate import odeint

def lanchester_square(state, t, alpha, beta):
    """확장 Lanchester 제곱법칙 ODE"""
    B, R = state
    dBdt = -alpha * max(R, 0)
    dRdt = -beta * max(B, 0)
    return [dBdt, dRdt]

def resolve_combat(blue_strength, red_strength,
                   blue_efficiency, red_efficiency,
                   terrain_mod_blue=1.0, terrain_mod_red=1.0,
                   duration=1, noise_std=0.1):
    """1턴 교전 결과 산출"""
    alpha = red_efficiency * terrain_mod_red
    beta = blue_efficiency * terrain_mod_blue

    t = np.linspace(0, duration, 10)
    solution = odeint(lanchester_square, [blue_strength, red_strength],
                      t, args=(alpha, beta))

    blue_final = solution[-1, 0]
    red_final = solution[-1, 1]

    # 확률적 노이즈 추가
    blue_noise = np.random.normal(0, noise_std * abs(blue_strength - blue_final))
    red_noise = np.random.normal(0, noise_std * abs(red_strength - red_final))

    blue_result = max(0, blue_final + blue_noise)
    red_result = max(0, red_final + red_noise)

    return {
        "blue_remaining": round(blue_result),
        "red_remaining": round(red_result),
        "blue_losses": blue_strength - round(blue_result),
        "red_losses": red_strength - round(red_result)
    }
```

### B. LLM Agent 추론 래퍼 (MLX 버전)

```python
from mlx_lm import load, generate

class LocalLLMAgent:
    def __init__(self, model_path, role_prompt, max_tokens=512):
        self.model, self.tokenizer = load(model_path)
        self.role_prompt = role_prompt
        self.max_tokens = max_tokens

    def decide(self, situation_text: str) -> dict:
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.role_prompt},
                {"role": "user", "content": situation_text}
            ],
            tokenize=False, add_generation_prompt=True
        )

        response = generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=0.7
        )

        return self._parse_json(response)

    def _parse_json(self, text: str) -> dict:
        """JSON 추출 및 파싱 (실패 시 기본 행동 반환)"""
        import json, re
        # JSON 블록 추출 시도
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # 파싱 실패 시 기본 방어 행동
        return {"reasoning": "Parse failure - default defense",
                "actions": [{"unit": "all", "action_type": "defend",
                            "target_hex": "current", "posture": "defense"}]}
```
