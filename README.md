# Multi-Agent Wargame

Python 3.11+ 기반의 전술 워게임 연구용 저장소입니다. 현재 목표는 "LLM 기반 지휘 에이전트"와 "스크립트/룰/랜덤 기반 베이스라인"을 동일한 엔진, 동일한 로그 포맷, 동일한 분석 파이프라인 위에서 비교할 수 있는 실험 프레임워크를 만드는 것입니다.

이 저장소는 `Research_plan.md`를 모듈 경계의 기준으로 사용하며, 다음 원칙을 유지합니다.

- `src` 레이아웃 패키지 구조
- `dataclass`, `Enum`, type hint 중심 설계
- 엔진은 LLM 구현체에 직접 의존하지 않음
- LangChain 미사용
- 실험 지표는 숨겨진 내부 상태가 아니라 저장된 JSONL 로그에서 계산

## 현재 구현된 범위

### 1. Tactical Map Layer

`wargame.core`

- `HexGrid`
  - axial hex coordinate 시스템
  - 이웃 hex 조회
  - hex distance 계산
- `TerrainLibrary`
  - 지형별 이동 비용과 방어 modifier 정의
  - 기본값은 명시적 상수 테이블로 관리
- 주요 dataclass
  - `Position`
  - `Unit`
  - `Force`
  - `Observation`
  - `ActionCommand`
  - `CombatResult`
  - `TurnResult`
  - `GameState`

핵심 의도는 "작고 자주 쓰이는 값 객체는 단순하고 읽기 쉽게", "엔진이 다루는 상태는 구조적으로 분명하게" 유지하는 것입니다.

### 2. Combat Resolver

`wargame.combat`

- deterministic / stochastic Lanchester-style attrition
- terrain defense modifier 반영
- 재현 가능한 stochastic 실행을 위한 `seed` 또는 `rng` 지원
- 음수 잔존 전력 방지
- one-turn combat result를 구조화된 dataclass로 반환

현재 모델은 1턴 단위의 단순 attrition 근사이며, 사기, 보급, 연속시간 적분, 사거리 세부 모델은 아직 포함하지 않습니다.

### 3. Engine and Fog of War

`wargame.engine`

- `StateManager`
  - canonical full state 관리
  - 안전한 snapshot 반환
  - 턴 메타데이터 관리
- `FogOfWarFilter`
  - faction-specific observation 생성
  - visibility / identification range 지원
  - uncertainty range를 포함한 machine-friendly 관측 정보 제공
- `SimulationEngine`
  - 액션 실행
  - 전투 해석 연결
  - terminal condition 판단
  - 턴 결과 기록

중요한 설계 제약은 엔진이 프롬프트 텍스트나 모델 포맷을 모른다는 점입니다. 엔진은 오직 typed state와 typed action만 처리합니다.

### 4. State-to-Text and Action Bridge

`wargame.agents.state_to_text`, `wargame.agents.parser`

- `StateRenderer`
  - faction view state를 deterministic tactical report 텍스트로 렌더링
- `ActionParser`
  - strict JSON schema 검증
  - `reasoning`, `doctrine_reference`, `actions[]` 구조 파싱
  - unit id, action type, posture, target hex 검증
  - LLM 출력 오류 복원 로직 (6종):
    - non-HOLD + `target_hex=null` → HOLD/DEFEND로 자동 강등 (전체 플랜 거부 없음)
    - HOLD + `target_hex≠null` → target_hex 자동 클리어
    - `action_type` 별칭 매핑: `maneuver→move`, `defend→hold`, `fire→support_by_fire`
    - `posture` 별칭 매핑: `offense→attack`, `observe→maneuver`, `recon→maneuver`
    - 적 팩션 unit_id 자동 드롭, suffix 부분 매칭 (`"assault"→"blue-assault"`)
    - 누락된 아군 유닛 자동으로 HOLD/DEFEND fallback 삽입
- malformed output 또는 invalid action 시 safe fallback plan 생성

fallback은 조용히 실패를 숨기지 않고, defensive hold로 바꾸면서 에러 정보를 metadata에 남깁니다.

### 5. Local LLM Wrapper Layer

`wargame.agents.local_llm`, `wargame.agents.prompts`

- `BaseAgent`
  - 모든 에이전트의 공통 인터페이스
- `LocalLLMAgent`
  - backend adapter + prompt registry + parser를 묶는 wrapper
- `LocalLLMBackend`
  - 모델 추론 구현체용 추상 인터페이스
- `ChatTemplateAdapter`
  - 모델 family별 chat template 분리
- `PromptRegistry`
  - `blue`, `red`, `white` 역할별 prompt 관리
- `MockLocalLLMBackend`
  - 실제 모델 로딩 없이 테스트 가능한 fake backend

이 구조는 MLX on Mac, Colab 환경의 다른 backend, 그리고 Qwen/Mistral/Llama 계열 wrapper를 나중에 붙이기 쉽게 설계되어 있습니다.

시스템 프롬프트는 FM 3-90 교리 기반으로 구성됩니다.

- `BLUE_SYSTEM_PROMPT`: FM 3-90 집중·기습·경계·기동 4원칙 + MDMP 3단계(ASSESS/DEVELOP/DECIDE)
- `RED_SYSTEM_PROMPT`: 종심방어·역습·지형활용·기만 4원칙 + MDMP 3단계
- `WHITE_CELL_SYSTEM_PROMPT`: 6개 교리 원칙 체크리스트 기반 판정
- `OUTPUT_CONTRACT`: 모든 역할의 JSON 출력 규약 명시 (unit_id는 아군 ID만 사용 강제)

### 6. Baseline Agents

`wargame.agents`

- `ScriptAgent`
  - `FRONTAL_ASSAULT`
  - `FLANK_MANEUVER`
  - `DELAY_DEFENSE`
- `RuleBasedAgent`
  - 해석 가능한 if-then rule 집합
  - 고정 seed 하에서 deterministic tie-breaking
- `RandomAgent`
  - valid action만 샘플링
  - 고정 seed 기반 재현 가능

세 베이스라인 모두 LLM 경로와 동일한 `ActionCommand` 포맷을 사용하므로 동일한 orchestrator와 로그 체계에서 비교할 수 있습니다.

### 7. Orchestration and Logging

`wargame.orchestrator`, `wargame.logging`

- `TurnLoop`
  - blue observation
  - blue decision
  - blue validation / execution
  - red observation
  - red decision
  - red validation / execution
  - combat resolution
  - turn advancement
  - structured logging
  - optional white-cell evaluation
  - terminal-state check
- `JsonlLogger`
  - turn-by-turn JSONL 기록
  - run-level context 포함

현재 로그에는 다음과 같은 분석용 정보가 남습니다.

- run context
  - `run_id`
  - `scenario_id`
  - `scenario_name`
  - `seed`
  - `blue_agent`
  - `red_agent`
  - `white_cell`
  - `initial_force_totals`
  - `metadata.seed_control`
- turn result
  - `turn`
  - `actions`
  - `combat`
  - `notes`
  - `metadata.blue`
  - `metadata.red`
  - `metadata.white_cell`
  - `state`

LLM 관련 의사결정 메타데이터에는 다음 필드가 들어갈 수 있습니다.

- `decision_source`
- `model_name`
- `role`
- `json_parse_success`
- `inference_time_s`
- `used_fallback`
- `error_type`
- `error_stage`

### 8. Experiments and Analysis

`wargame.experiments`, `wargame.analysis`

- `ExperimentRunner`
  - 단일 run 실행
  - run context를 logger에 주입
  - `prepare_run` hook으로 seed 적용 또는 RNG 재초기화 가능
- `BatchRunner`
  - scenario x matchup x seed matrix 확장
  - `runner_factory`로 프로젝트별 engine/agent 조립을 위임
- seed utilities
  - `DEFAULT_SEEDS`
  - `get_seed_sequence`
- 로그 기반 분석 지표
  - `doctrine_compliance_rate`
  - `tactical_rationality_score`
  - `action_entropy`
  - `win_rate`
  - `mean_remaining_force_ratio`
  - `escalation_sensitivity_index`
  - `tactic_transition_frequency`
  - `json_parsing_success_rate`
  - `inference_time_summary`
  - `summarize_runs`

`doctrine_compliance_rate`와 `tactical_rationality_score`는 현재 white-cell 또는 hook 기반 placeholder 집계 인터페이스입니다. 실제 domain-specific scoring logic은 이후 단계에서 추가할 예정입니다.

## 현재 권장 실험 단위

이 저장소에서 실험의 기본 단위는 아래 조합입니다.

`scenario x blue agent x red agent x seed`

한 run은 다음 순서로 실행됩니다.

1. scenario / initial state 구성
2. engine 조립
3. blue / red / optional white-cell agent 조립
4. `TurnLoop` 실행
5. 각 turn을 JSONL로 저장
6. 저장된 JSONL 로그를 분석 함수로 집계

즉, 실험 결과는 "엔진 내부 객체"가 아니라 "저장된 turn log"를 중심으로 다루는 것이 기본입니다.

## 재현성 원칙

현재 구현에서 재현성은 아래처럼 다룹니다.

- stochastic combat은 `seed` 또는 `rng`를 직접 받을 수 있습니다.
- `RandomAgent`와 일부 baseline helper는 seed 기반으로 재현 가능합니다.
- `RuleBasedAgent`는 고정 seed 하에서 deterministic tie-breaking을 사용합니다.
- `ExperimentRunner.seed`는 선언만으로 재현성을 보장하지 않습니다.
- 실제 재현성을 확보하려면 `prepare_run(turn_loop, seed)` hook에서 combat resolver, agent RNG, 기타 stochastic component를 명시적으로 재초기화해야 합니다.

즉, `seed`는 "로그에 남는 실험 조건"이고, `prepare_run`은 "실제로 그 조건을 시스템에 적용하는 지점"입니다.

## Quick Start

```bash
python -m pip install -e .[dev]
pytest
```

직접 Python에서 사용할 때는 editable install 이후 일반 import를 쓰면 됩니다.

## Local LLM 실험 설정

로컬 LLM 실험은 실행 환경에 따라 설치와 CLI 인자가 조금 다릅니다.

```bash
# Mac Apple Silicon + MLX
python -m pip install -e ".[dev,analysis,llm-mlx]"

# Linux / Colab + vLLM
python -m pip install -e ".[dev,analysis]"
python -m pip install vllm
```

`pyproject.toml`의 `llm-mlx` extra는 `mlx-lm`을 macOS arm64 환경에서만 설치하도록 정의되어 있습니다.

CLI 에이전트 스펙은 아래 형식을 사용합니다.

- baseline: `rule`, `random`, `script`, `script:frontal_assault`, `script:flank_maneuver`, `script:delay_defense`
- local LLM: `local_llm:<model_id_or_path>`

예시는 다음과 같습니다.

```bash
# Mac M-series / MLX
python scripts/run_single_game.py \
  --scenario s1_open_encounter \
  --blue-agent local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit \
  --red-agent rule \
  --backend mlx \
  --visibility-radius 5 \
  --identification-radius 2 \
  --output runs/qwen_s1.jsonl

# Colab / vLLM
python scripts/run_batch.py \
  --scenario s1_open_encounter \
  --matchup "local_llm:mistralai/Mistral-7B-Instruct-v0.3,rule" \
  --backend vllm \
  --seed-count 10 \
  --visibility-radius 5 \
  --identification-radius 2 \
  --output-dir runs/mistral_batch
```

Fog-of-war와 max-tokens 권장 설정은 아래와 같습니다.

```bash
# fog-preset 옵션 (--visibility-radius / --identification-radius 대신 사용 가능)
--fog-preset baseline   # vr=8, idr=3 (Rule/Script/Random 베이스라인 비교용)
--fog-preset llm        # vr=5, idr=2 (LLM 실험용 — 제한적 정보 조건)

# max-tokens 권장값 (JSON 중단 방지)
--max-tokens 1024       # 일반 실험 권장
--max-tokens 4096       # 긴 reasoning 또는 s2+ 복잡 시나리오용
```

> **주의**: Phase 3 베이스라인은 `fog-preset baseline`, Phase 4 LLM 실험은 `fog-preset llm` 조건으로 실행됩니다. 두 실험을 직접 비교할 때는 fog 조건 차이를 고려해야 합니다.

## 단일 실험 예시

아래 예시는 baseline agent끼리 1턴짜리 실험을 실행하고 JSONL 로그를 남기는 최소 예시입니다.

```python
from pathlib import Path

from wargame.agents import ActionParser, RandomAgent, RuleBasedAgent, StateRenderer
from wargame.combat import LanchesterResolver
from wargame.core import Faction, GameState, HexGrid, Position, TerrainType, Unit
from wargame.engine import FogOfWarFilter, SimulationEngine, StateManager
from wargame.experiments import ExperimentRunner
from wargame.logging import JsonlLogger
from wargame.orchestrator import TurnLoop
from wargame.scenarios.schema import ScenarioSpec

grid = HexGrid(width=5, height=5)
initial_state = GameState(
    turn=0,
    max_turns=1,
    units={
        "blue-1": Unit(
            unit_id="blue-1",
            faction=Faction.BLUE,
            position=Position(0, 0),
            strength=100,
        ),
        "red-1": Unit(
            unit_id="red-1",
            faction=Faction.RED,
            position=Position(1, 0),
            strength=100,
        ),
    },
    terrain_by_hex={
        Position(0, 0): TerrainType.OPEN,
        Position(1, 0): TerrainType.OPEN,
    },
)

fog = FogOfWarFilter(visibility_radius=3, identification_radius=1)
engine = SimulationEngine(
    state_manager=StateManager(initial_state=initial_state, fog_of_war=fog),
    combat_resolver=LanchesterResolver(),
    fog_of_war=fog,
)

turn_loop = TurnLoop(
    engine=engine,
    blue_agent=RuleBasedAgent(grid=grid, faction=Faction.BLUE, seed=0),
    red_agent=RandomAgent(grid=grid, faction=Faction.RED, seed=0),
    renderer=StateRenderer(),
    parser=ActionParser(grid=grid),
    logger=JsonlLogger(path=Path("runs/demo.jsonl")),
)

runner = ExperimentRunner(
    scenario=ScenarioSpec(scenario_id="demo", name="Demo Scenario"),
    turn_loop=turn_loop,
    seed=0,
    prepare_run=lambda loop, seed: None,
)

run = runner.run()
print(run.log_path)
```

## 배치 실험 방법

`BatchRunner`는 matrix expansion만 담당하고, 실제 engine/agent 조립은 `runner_factory`에 위임합니다. 이 방식은 scenario별 초기 상태 구성 방식이나 backend 조합 방식이 아직 실험마다 달라질 수 있기 때문입니다.

개념적으로는 아래 순서로 사용합니다.

```python
from pathlib import Path

from wargame.experiments import BatchRunner, MatchupSpec, get_seed_sequence


def runner_factory(condition):
    # scenario + matchup + seed를 받아
    # TurnLoop / Logger / ExperimentRunner를 구성해 반환
    return build_experiment_runner(condition)


batch = BatchRunner.from_matrix(
    scenarios=[scenario_a, scenario_b],
    matchups=[
        MatchupSpec(
            name="rule_vs_random",
            blue_agent_name="rule",
            red_agent_name="random",
        ),
        MatchupSpec(
            name="script_vs_rule",
            blue_agent_name="script",
            red_agent_name="rule",
        ),
    ],
    seeds=get_seed_sequence(5),
    runner_factory=runner_factory,
    output_dir=Path("runs/batch"),
)

results = batch.run()
```

## 로그 기반 분석 방법

실험 분석은 JSONL 파일 또는 이미 메모리에 로드된 record list 둘 다 받을 수 있습니다.

```python
from wargame.analysis import (
    action_entropy,
    escalation_sensitivity_index,
    inference_time_summary,
    json_parsing_success_rate,
    load_jsonl_records,
    mean_remaining_force_ratio,
    summarize_runs,
    tactic_transition_frequency,
    win_rate,
)

records = load_jsonl_records("runs/demo.jsonl")

print(action_entropy(records))
print(escalation_sensitivity_index(records))
print(tactic_transition_frequency(records))
print(json_parsing_success_rate(records))
print(inference_time_summary(records))

run_paths = ["runs/demo_0.jsonl", "runs/demo_1.jsonl", "runs/demo_2.jsonl"]
print(win_rate(run_paths))
print(mean_remaining_force_ratio(run_paths))
print(summarize_runs(run_paths))
```

### 현재 제공되는 지표 해석

- `action_entropy`
  - action type 분포의 Shannon entropy
  - 행동 다양성이 높을수록 커집니다.
- `win_rate`
  - 최종 logged state 또는 combat winner를 바탕으로 계산합니다.
- `mean_remaining_force_ratio`
  - 초기 병력 대비 최종 잔존 병력 비율입니다.
- `escalation_sensitivity_index`
  - turn-to-turn aggression score 변화량의 평균입니다.
- `tactic_transition_frequency`
  - 동일 unit가 연속 turn 사이에 action type을 얼마나 자주 바꾸는지 봅니다.
- `json_parsing_success_rate`
  - LLM decision metadata의 `json_parse_success`를 집계합니다.
- `inference_time_summary`
  - turn metadata에 기록된 추론 시간 통계를 집계합니다.

## 테스트 범위

현재 테스트는 아래 계층을 포함합니다.

- hex grid / terrain / models
- Lanchester combat
- fog of war / state snapshot
- state-to-text / action parser
- local LLM wrapper and fake backend
- baseline agents
- turn loop integration
- experiment metrics

## 실험 현황 (2026-03-25 기준)

| Phase | 내용 | 상태 | 게임 수 |
|---|---|---|---|
| Phase 1 | 엔진 검증 (단위 테스트 + 시나리오 검증) | ✅ 완료 | 38개 테스트 PASS |
| Phase 2 | LLM 구축 + 안정성 테스트 | ✅ 완료 | fallback ~14% 달성 |
| Phase 3 | Rule-vs-Rule 베이스라인 | ✅ 완료 | 250 게임 (5시나리오 × 50 seeds) |
| Phase 4-1 | Qwen2.5-7B (Mac M4, MLX) | ⏳ 진행중 | 48/100 게임 완료 |
| Phase 4-2 | Mistral-7B (Colab, vLLM) | ⏳ 로컬 전송 대기 | 완료 |
| Phase 4-3 | Llama-3.1-8B (Colab, vLLM) | ⏳ 로컬 전송 대기 | 완료 |
| Phase 5 | 통계 분석 + 논문 | ⏳ 대기 | — |

**Phase 3 베이스라인 집계 결과** (Rule vs Rule, 250 게임):

- DCR (교리 준수율): **0.804**
- TRS (전술 합리성): **3.548** / 5
- Action Entropy: **1.620**
- ESI (공격성 변동성): **0.102**

**Phase 4 Qwen 초기 결과** (진행중):

- S1 평지 조우전: Blue 승률 **~20%** (베이스라인 6% 대비 +14%p)
- S2 산악 방어진지: Blue 승률 **~0%** (베이스라인 ~96% 대비 -96%p) — unit_id hallucination 이슈

## 아직 placeholder인 부분

아래 영역은 구조는 만들어져 있지만 구현이 아직 얕거나 placeholder입니다.

- white-cell의 실제 doctrine / rationality scoring
- plot/report artifact의 추가 고도화
- `state_to_text.py`: unit_id 힌트 라인 추가 예정 (unit_id hallucination 방지)

즉, 현재 저장소는 "핵심 엔진 vertical slice + 비교 가능한 베이스라인 + JSONL 기반 실험 실행기 + 로그 기반 분석 기초"까지는 준비되어 있고, 평가지표 정교화와 시각화 고도화는 이어서 확장하는 단계입니다.
