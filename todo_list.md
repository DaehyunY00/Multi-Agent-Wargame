# 실험 진행 체크리스트 — LLM 기반 다중 에이전트 워게임

> **진행 규칙**
> - 각 단계 완료 시 `[ ]` → `[x]`로 변경
> - `# 결과 확인` 블록의 기준을 **통과해야만** 다음 단계로 진행
> - 실험 로그는 모두 `runs/` 하위에 저장, `.gitignore`에서 관리
> - 컴퓨팅 환경: **Mac M4** (Python 3.13, mlx-lm) / **Colab Pro A100** (vLLM)

> **⚠️ 중요 파라미터 규칙 — Phase 1 검증에서 발견 (2차 수정 반영)**
>
> | 실험 목적 | `--visibility-radius` | `--identification-radius` | 이유 |
> |---|---|---|---|
> | 베이스라인 (rule/random/script) | **8** | **3** | 초기 배치 거리(6~7) 커버 + 위치 확인 보장 |
> | LLM 에이전트 실험 | **5** | **2** | 불확실성 유지 + 교전 발생 균형 |
>
> **1차 발견 (idr 문제)**: `identification_radius=1(기본값)`이면 적 위치가 `None` → `RECON` 반복 → 교전 미발생.
>
> **2차 발견 (vr 문제)**: `identification_radius=3`으로 올려도 `visibility_radius=3(기본값)`이면
> 초기 배치 거리(6~7)가 감지 범위 밖이라 `enemy_observations` 자체가 비어있음 → 동일 증상.
> `--visibility-radius 8`로 올린 뒤 s1 T04에서 교전 발생 확인 (Blue -8, Red -3) ✅
>
> **combat 필드 구조 주의**: 손실 확인 시 `combat.blue_loss` 키가 아닌
> `combat.casualties_by_unit` 딕셔너리 또는 `combat.summary` 문자열을 사용해야 함.

---

## 환경 설정

```bash
# 최초 1회만 실행
cd Multi-Agent_Wargame
pip install -e ".[dev,analysis,llm-mlx]"
mkdir -p logs runs

# Colab 환경 (vLLM 사용 시)
# pip install -e ".[dev,analysis]"
# pip install vllm
```

> **optional dependencies 그룹 설명** (pyproject.toml 기준)
>
> | 그룹 | 포함 패키지 | 용도 |
> |---|---|---|
> | `dev` | pytest | 단위 테스트 |
> | `analysis` | scipy, numpy | 통계 분석 |
> | `llm-mlx` | mlx-lm | Mac M4 로컬 추론 |
>
> `matplotlib`은 `plots` 그룹에 포함. 시각화 필요 시 `pip install -e ".[dev,analysis,llm-mlx,plots]"`
> `llm-mlx`는 pyproject 기준으로 macOS arm64에서만 `mlx-lm`을 설치함.

### 확인 명령어

```bash
python -m pytest tests/ -v 2>&1 | tee logs/pytest_setup.txt
```

#### 결과 확인

- [x] 전체 테스트 **38/38 PASS** ✅ (Python 3.13.5 환경 확인 완료)
- [x] `tests/test_lanchester.py` — 5개 항목 통과
- [x] `tests/test_hexgrid.py` — 통과
- [x] `tests/test_fog_of_war.py` — 통과
- [x] `tests/test_simulation_engine.py` — 통과
- [x] `tests/test_turn_loop.py` — 통과

---

## Phase 1 — 시뮬레이션 엔진 검증 (Week 1–2)

### 1-1. 단위 테스트 전체 실행

```bash
mkdir -p logs
python -m pytest tests/ -v --tb=short 2>&1 | tee logs/pytest_phase1.txt
```

#### 결과 확인

- [x] 전체 통과율 **100%** (38/38) ✅
- [x] `logs/pytest_phase1.txt` 저장 확인 ✅

---

### 1-2. 단일 게임 동작 확인 (Rule vs Rule, vr=8 / idr=3 적용)

> ⚠️ `--visibility-radius 8 --identification-radius 3` 모두 필수
> — vr=3(기본)이면 초기 배치 거리(6~7)가 감지 범위 밖이라 교전 미발생

```bash
mkdir -p runs/phase1

python scripts/run_single_game.py \
  --scenario s1_open_encounter \
  --blue-agent rule \
  --red-agent rule \
  --visibility-radius 8 \
  --identification-radius 3 \
  --output runs/phase1/s1_rule_vr8_idr3.jsonl
```

#### 결과 확인

- [x] 종료 메시지에 `"terminal": true` 포함 ✅
- [x] `runs/phase1/s1_rule_vr8_idr3.jsonl` 파일 생성 ✅
- [x] `"turns"` 값이 8~20 범위 내 ✅ (turns=12)
- [x] 교전 발생 확인 ✅ — T04에서 교전 1회 (Blue -8, Red -3), 최종 Blue=292, Red=297

---

### 1-3. 시나리오 5종 × Rule 에이전트 동작 확인 (vr=8 / idr=3 적용)

```bash
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_single_game.py \
    --scenario $SCENARIO \
    --blue-agent rule \
    --red-agent rule \
    --visibility-radius 8 \
    --identification-radius 3 \
    --output runs/phase1/${SCENARIO}_rule_vr8_idr3.jsonl
  echo "Done: $SCENARIO"
done

# 교전 발생 여부 확인 (올바른 combat 필드 키 사용)
python -c "
import json, pathlib
for s in ['s1_open_encounter','s2_mountain_assault','s3_urban_fight','s4_river_crossing','s5_breakout']:
    p = pathlib.Path(f'runs/phase1/{s}_rule_vr8_idr3.jsonl')
    if not p.exists(): print(f'{s}: 파일 없음'); continue
    records = [json.loads(l) for l in p.read_text().strip().split('\n') if l]
    last = records[-1]
    units = last.get('state', {}).get('units', {})
    blue = sum(u['strength'] for u in units.values() if u.get('faction')=='blue')
    red  = sum(u['strength'] for u in units.values() if u.get('faction')=='red')
    combat_turns = sum(1 for r in records if r.get('combat') and r['combat'].get('casualties_by_unit'))
    print(f'{s}: {len(records)}턴, 교전={combat_turns}턴, Blue={blue}, Red={red}')
"
```

#### 결과 확인 — 기존(vr=3/idr=1) vs vr=8/idr=3 비교

| 시나리오 | 기존 교전 | 기존 결과 | vr=8/idr=3 교전 | vr=8/idr=3 결과 |
|---|---|---|---|---|
| s1_open_encounter | 0/12턴 ⚠️ | DRAW (손실 0) | **1/12턴 ✅** | DRAW (Blue=292, Red=297) |
| s2_mountain_assault | 0/14턴 ⚠️ | DRAW (손실 0) | **0/14턴 ⚠️** | DRAW (Blue=325, Red=310) |
| s3_urban_fight | 10/12턴 ✅ | RED 승 (B:228, R:235) | **8/12턴 ✅** | RED 승 (Blue=261, Red=279) |
| s4_river_crossing | 0/14턴 ⚠️ | DRAW (손실 0) | **0/14턴 ⚠️** | DRAW (Blue=305, Red=300) |
| s5_breakout | 5/13턴 ✅ | BLUE 승 (B:278, R:253) | **5/13턴 ✅** | BLUE 승 (Blue=278, Red=253) |

> ⚠️ **s2·s4 교전 미발생 원인 (vr=8에서도 동일)**: 감지는 되지만 Rule 에이전트가 ATTACK을 선택하지 않음.
> s2(산악): 지형 불리 → Blue가 열세 판정 후 WITHDRAW 반복.
> s4(하천): 도하 취약 상황 → 양측 HOLD 위주. **시나리오 설계 의도에 맞는 Rule 에이전트 동작**으로 기록.
> (LLM 에이전트는 더 공격적 의사결정이 가능하므로 Phase 4에서 차이 관찰 예정)

- [x] s1 vr=8/idr=3 교전 발생 ✅ (1턴, Blue -8, Red -3)
- [x] s3 vr=8/idr=3 교전 발생 ✅ (8턴, Blue=261, Red=279)
- [x] s5 vr=8/idr=3 교전 발생 ✅ (5턴, Blue=278, Red=253)
- [x] s2·s4 교전 없음 — Rule 에이전트 WITHDRAW/HOLD 선택, 시나리오 정상 동작으로 기록 ✅
- [x] 모든 게임 `terminal: true` ✅

---

### 1-4. Lanchester 확률적 모드 확인 (vr=8 / idr=3 적용)

> vr=8/idr=3 환경에서 교전이 발생해야 stochastic 노이즈 효과를 확인할 수 있음.

```bash
python scripts/run_single_game.py \
  --scenario s1_open_encounter \
  --blue-agent rule --red-agent rule \
  --visibility-radius 8 --identification-radius 3 \
  --output runs/phase1/s1_deterministic_vr8_idr3.jsonl

python scripts/run_single_game.py \
  --scenario s1_open_encounter \
  --blue-agent rule --red-agent rule \
  --visibility-radius 8 --identification-radius 3 \
  --stochastic-combat --noise-std 0.1 \
  --output runs/phase1/s1_stochastic_vr8_idr3.jsonl
```

> ⚠️ **zsh 주의**: `#` 주석이 포함된 여러 명령을 한 번에 붙여넣으면
> `zsh: command not found: #` 오류가 발생함. 위 두 명령을 **각각 별도로** 실행할 것.

```bash
python -c "
import json, pathlib
for label, path in [
    ('deterministic', 'runs/phase1/s1_deterministic_vr8_idr3.jsonl'),
    ('stochastic',    'runs/phase1/s1_stochastic_vr8_idr3.jsonl'),
]:
    records = [json.loads(l) for l in pathlib.Path(path).read_text().strip().split('\n') if l]
    last = records[-1]
    units = last.get('state', {}).get('units', {})
    blue = sum(u['strength'] for u in units.values() if u.get('faction') == 'blue')
    red  = sum(u['strength'] for u in units.values() if u.get('faction') == 'red')
    combat_turns = sum(1 for r in records if r.get('combat') and r['combat'].get('casualties_by_unit'))
    print(f'{label}: Blue={blue}, Red={red}, 교전={combat_turns}턴')
"
```

#### 결과 확인

> **현재 상태 (2026-03-21)**:
> - 결정론적 실행 → `s1_deterministic_vr8_idr3.jsonl` ✅ 생성 완료
> - 확률적 실행 → `s1_stochastic_vr8_idr3.jsonl` ✅ 생성 완료 (turns=12)
> - **비교 스크립트 아직 미실행** → 아래 python -c 명령어를 실행해야 함

- [x] 결정론적 실행 완료 (`s1_deterministic_vr8_idr3.jsonl`) ✅
- [x] 확률적 실행 완료 (`s1_stochastic_vr8_idr3.jsonl`) ✅
- [ ] 두 실행의 최종 잔존 병력이 **다름** (stochastic 노이즈 효과 확인) ← **비교 스크립트 실행 필요**
- [ ] 오류 없이 종료

---

## Phase 2 — LLM 에이전트 시스템 구축 (Week 3–4)

### 2-1. MLX 환경 구축 (Mac M4)

> MLX 백엔드 (`src/wargame/agents/local_llm.py`) 구현 완료 ✅

```bash
# llm-mlx 그룹이 포함된 설치 (환경 설정 섹션에서 이미 완료)
pip install -e ".[dev,analysis,llm-mlx]"

# Qwen2.5-7B-Instruct 4bit 양자화 모델 다운로드
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-7B-Instruct-4bit')"
```

#### 결과 확인

- [x] `mlx-lm` import 오류 없음 ✅
- [ ] 모델 파일 로컬 캐시 확인 (`~/.cache/huggingface/hub/`)
- [ ] 추론 속도 벤치마크: **50 tok/s 이상**

```bash
# 속도 벤치마크
python -c "
from mlx_lm import load, generate
import time
model, tokenizer = load('mlx-community/Qwen2.5-7B-Instruct-4bit')
prompt = 'You are a military commander. What is your next move?'
t0 = time.time()
out = generate(model, tokenizer, prompt=prompt, max_tokens=200)
elapsed = time.time() - t0
print(f'tok/s: {200/elapsed:.1f}')
"
```

---

### 2-2. JSON 출력 안정성 검증 — 단일 게임 10회 (Qwen2.5-7B)

> CLI LLM 스펙 (`--blue-agent local_llm:<HF_MODEL_ID>`) 구현 완료 ✅
>
> LLM 에이전트: `--visibility-radius 5 --identification-radius 2` 적용 (불확실성 유지 + 교전 균형)

```bash
mkdir -p runs/phase2/llm_stability

# 단일 게임 실행 (안정성 확인용)
python scripts/run_single_game.py \
  --scenario s1_open_encounter \
  --blue-agent local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit \
  --red-agent rule \
  --visibility-radius 5 \
  --identification-radius 2 \
  --output runs/phase2/llm_stability/qwen_test_01.jsonl

# 파싱 성공률 확인
python -c "
import json, pathlib
path = pathlib.Path('runs/phase2/llm_stability/qwen_test_01.jsonl')
records = [json.loads(l) for l in path.read_text().strip().split('\n') if l]
fallbacks = sum(1 for r in records if r.get('metadata', {}).get('blue', {}).get('used_fallback'))
print(f'총 {len(records)} 턴, 폴백 {fallbacks}회 ({100*fallbacks/max(len(records),1):.1f}%)')
"
```

#### 결과 확인

- [ ] JSON 파싱 성공률 ≥ **90%** (used_fallback < 10%)
- [ ] 비정상 종료 없음

---

### 2-3. 프롬프트 반복 안정성 검증 — 10회 배치 (Qwen2.5-7B)

```bash
for i in $(seq 1 10); do
  python scripts/run_single_game.py \
    --scenario s2_mountain_assault \
    --blue-agent local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit \
    --red-agent rule \
    --visibility-radius 5 \
    --identification-radius 2 \
    --seed $i \
    --output runs/phase2/llm_stability/prompt_test_s${i}.jsonl
done

# 10회 통계 요약
python -c "
import json, pathlib
from wargame.analysis.metrics import action_entropy, json_parsing_success_rate

logs = sorted(pathlib.Path('runs/phase2/llm_stability').glob('prompt_test_s*.jsonl'))
entropies = [action_entropy(p) for p in logs]
parse_rates = [json_parsing_success_rate(p) for p in logs]
print(f'Action Entropy: mean={sum(entropies)/len(entropies):.3f} bits')
print(f'Parse Success : mean={sum(parse_rates)/len(parse_rates):.3f}')
"
```

#### 결과 확인

- [ ] 10회 중 오류 종료 **0회**
- [ ] `mean_action_entropy` > 0 (결정론적이지 않음)
- [ ] `mean_json_parsing_success_rate` ≥ 0.90

---

### 2-4. Colab 환경 구성 (Mistral-7B, Llama-3.1-8B)

> Colab에서 실행 — 아래 명령어를 Colab 노트북 셀에 붙여넣기

```bash
# [Colab 셀 1] 의존성 설치
!pip install vllm
!git clone https://github.com/<your_repo>/Multi-Agent_Wargame.git
%cd Multi-Agent_Wargame
!pip install -e ".[dev,analysis]"

# [Colab 셀 2] Mistral-7B 로드 확인
!python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.3', quantization='bitsandbytes', load_format='bitsandbytes')
params = SamplingParams(temperature=0.7, max_tokens=50)
out = llm.generate(['Hello'], params)
print('Mistral 로드 완료:', out[0].outputs[0].text[:80])
"

# [Colab 셀 3] Llama-3.1-8B 로드 확인
!python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='meta-llama/Llama-3.1-8B-Instruct')
params = SamplingParams(temperature=0.7, max_tokens=50)
out = llm.generate(['Hello'], params)
print('Llama 로드 완료:', out[0].outputs[0].text[:80])
"
```

#### 결과 확인

- [ ] Colab A100에서 vLLM 설치 완료
- [ ] Mistral-7B 로드 성공, 추론 속도 **70 tok/s 이상**
- [ ] Llama-3.1-8B 로드 성공, 추론 속도 **70 tok/s 이상**

---

## Phase 3 — 베이스라인 시스템 검증 (Week 5)

### 3-0. ✅ 사전 코드 수정 완료 — `run_batch.py` identification-radius 지원

> **[x] 완료** — `run_batch.py`에 `--visibility-radius` / `--identification-radius` 인자 추가 및
> `_build_runner` 내 `FogOfWarFilter` 호출 수정 완료.

수정 내용 요약 (참고용):
```python
# 1) argparse 인자 추가
parser.add_argument("--visibility-radius",     type=int, default=3)
parser.add_argument("--identification-radius",  type=int, default=1)

# 2) _build_runner 시그니처
def _build_runner(*, condition, white_cell_spec, stochastic_combat, noise_std,
                  visibility_radius=3, identification_radius=1): ...

# 3) FogOfWarFilter 호출
fog = FogOfWarFilter(
    visibility_radius=visibility_radius,
    identification_radius=identification_radius,
)

# 4) runner_factory 람다에서 전달
runner_factory=lambda condition: _build_runner(
    condition=condition, ...,
    visibility_radius=args.visibility_radius,
    identification_radius=args.identification_radius,
),
```

수정 후 동작 확인:
```bash
python scripts/run_batch.py \
  --scenario s1_open_encounter \
  --matchup rule,rule \
  --seed-count 3 \
  --identification-radius 3 \
  --output-dir runs/test_idr3/

python -c "
import json, pathlib
logs = list(pathlib.Path('runs/test_idr3').rglob('*.jsonl'))
print(f'로그 파일 수: {len(logs)}')
for p in logs[:3]:
    last = json.loads(p.read_text().strip().split('\n')[-1])
    print(f'  {p.name}: turns={last[\"turn\"]}, terminal={last.get(\"terminal\")}')
"
```

- [x] `--identification-radius` 인자 추가 및 코드 수정 완료 ✅
- [x] 소규모 배치(3회) 실행 후 교전 발생 확인 ✅

---

### 3-1. 베이스라인 단위 테스트

```bash
python -m pytest tests/test_baseline_agents.py -v
```

#### 결과 확인

- [ ] 전체 통과

---

### 3-2. 베이스라인 3종 × 시나리오 5종 × 50회 반복

> 베이스라인: `--identification-radius 3` 고정

```bash
mkdir -p runs/phase3/{rule,random,script}

# Rule-Based: 5 시나리오 × 50 seed
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup rule,rule \
    --seed-count 50 \
    --visibility-radius 8 \
    --identification-radius 3 \
    --output-dir runs/phase3/rule/${SCENARIO}
  echo "[Rule] Done: $SCENARIO"
done

# Random: 5 시나리오 × 50 seed
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup random,random \
    --seed-count 50 \
    --visibility-radius 8 \
    --identification-radius 3 \
    --output-dir runs/phase3/random/${SCENARIO}
  echo "[Random] Done: $SCENARIO"
done

# Script (Frontal Assault): 5 시나리오 × 50 seed
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "script:frontal_assault,script:frontal_assault" \
    --seed-count 50 \
    --visibility-radius 8 \
    --identification-radius 3 \
    --output-dir runs/phase3/script/${SCENARIO}
  echo "[Script] Done: $SCENARIO"
done
```

#### 결과 확인 (시나리오 균형성)

| 시나리오 | Rule Blue 승률 | Random Entropy | Script Entropy |
|---|---|---|---|
| s1_open_encounter | ___% | ___ bits | ___ bits |
| s2_mountain_assault | ___% | ___ bits | ___ bits |
| s3_urban_fight | ___% | ___ bits | ___ bits |
| s4_river_crossing | ___% | ___ bits | ___ bits |
| s5_breakout | ___% | ___ bits | ___ bits |

```bash
# 시나리오별 통계 출력
for AGENT in rule random script; do
  echo "=== $AGENT ==="
  python scripts/evaluate_logs.py runs/phase3/${AGENT}/
done
```

- [ ] 모든 시나리오 Blue 승률 **40~60%** 범위 내
- [ ] 교착(평균 게임 길이 = max_turns) 빈도 < 20%
- [ ] Script 에이전트 엔트로피 ≈ 0 (완전 결정론적)

---

### 3-3. 베이스라인 Action Entropy 비교 (RQ3 사전 확인)

```bash
python scripts/evaluate_logs.py \
  runs/phase3/rule/ \
  runs/phase3/random/ \
  runs/phase3/script/ \
  --plot-dir runs/phase3/plots/
```

#### 결과 확인

- [ ] `random` entropy > `rule` entropy > `script` entropy (**Script ≈ 0**)
- [ ] `plots/` 하위 SVG 파일 확인

---

## Phase 4 — 대규모 실험 (Week 6–8)

> **파라미터 구분**:
> - 베이스라인: `--identification-radius 3`
> - LLM 에이전트: `--identification-radius 2`

### 4-1. LLM 실험: Qwen2.5-7B × 5 시나리오 × 100회 (Mac M4)

```bash
mkdir -p runs/phase4/qwen

for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit,rule" \
    --seed-count 100 \
    --stochastic-combat \
    --noise-std 0.1 \
    --visibility-radius 5 \
    --identification-radius 2 \
    --output-dir runs/phase4/qwen/${SCENARIO} \
  && echo "[Qwen] Done: $SCENARIO"
done
```

> 예상 소요 시간: 시나리오당 약 2시간 (15턴 × 5초/턴 × 100회) = 총 10시간

#### 결과 확인

- [ ] 5개 시나리오 모두 완료 (jsonl 파일 500개)
- [ ] JSON 파싱 성공률 ≥ **90%**
- [ ] `used_fallback` 비율 < 10%

```bash
# 로그 무결성 확인
find runs/phase4/qwen/ -name "*.jsonl" | wc -l   # 기대값: 500

python -c "
import json, pathlib
errors = []
for p in pathlib.Path('runs/phase4/qwen').rglob('*.jsonl'):
    try:
        [json.loads(l) for l in p.read_text().strip().split('\n') if l]
    except Exception as e:
        errors.append((str(p), str(e)))
print(f'손상 파일: {len(errors)}개')
"
```

---

### 4-2. LLM 실험: Mistral-7B × 5 시나리오 × 100회 (Colab A100)

```bash
# [Colab 셀] 실험 실행
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "local_llm:mistralai/Mistral-7B-Instruct-v0.3,rule" \
    --backend vllm \
    --seed-count 100 \
    --stochastic-combat \
    --noise-std 0.1 \
    --visibility-radius 5 \
    --identification-radius 2 \
    --output-dir /content/drive/MyDrive/wargame_runs/mistral/${SCENARIO}
  echo "[Mistral] Done: $SCENARIO"
done
```

> 세션 단절 대비: `--seed-count 25`로 4 배치 분할 후 병합 가능

#### 결과 확인

- [ ] 5개 시나리오 모두 완료
- [ ] Google Drive 자동 저장 확인
- [ ] JSON 파싱 성공률 ≥ **90%**

---

### 4-3. LLM 실험: Llama-3.1-8B × 5 시나리오 × 100회 (Colab A100)

```bash
# [Colab 셀] 실험 실행
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "local_llm:meta-llama/Llama-3.1-8B-Instruct,rule" \
    --backend vllm \
    --seed-count 100 \
    --stochastic-combat \
    --noise-std 0.1 \
    --visibility-radius 5 \
    --identification-radius 2 \
    --output-dir /content/drive/MyDrive/wargame_runs/llama/${SCENARIO}
  echo "[Llama] Done: $SCENARIO"
done
```

#### 결과 확인

- [ ] 5개 시나리오 모두 완료
- [ ] JSON 파싱 성공률 ≥ **90%**

---

### 4-4. 베이스라인 비교 실험: 100회 반복 (Mac M4)

```bash
mkdir -p runs/phase4/baseline/{rule,random,script}

for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "rule,rule" \
    --seed-count 100 \
    --visibility-radius 8 \
    --identification-radius 3 \
    --output-dir runs/phase4/baseline/rule/${SCENARIO}

  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "random,random" \
    --seed-count 100 \
    --visibility-radius 8 \
    --identification-radius 3 \
    --output-dir runs/phase4/baseline/random/${SCENARIO}

  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "script:frontal_assault,script:frontal_assault" \
    --seed-count 100 \
    --visibility-radius 8 \
    --identification-radius 3 \
    --output-dir runs/phase4/baseline/script/${SCENARIO}
  echo "Done: $SCENARIO"
done
```

---

### 4-5. White Cell 배치 평가 및 전체 취합

```bash
python scripts/evaluate_logs.py runs/phase4/qwen/    --plot-dir runs/phase4/plots/qwen/
python scripts/evaluate_logs.py runs/phase4/mistral/ --plot-dir runs/phase4/plots/mistral/
python scripts/evaluate_logs.py runs/phase4/llama/   --plot-dir runs/phase4/plots/llama/
python scripts/evaluate_logs.py runs/phase4/baseline/ --plot-dir runs/phase4/plots/baseline/

# 전체 취합 요약
python scripts/evaluate_logs.py \
  runs/phase4/qwen/ runs/phase4/mistral/ runs/phase4/llama/ runs/phase4/baseline/ \
  > runs/phase4/summary_all.json
```

#### 결과 확인 — 논문 핵심 지표

| 지표 | Script | Rule | Random | Qwen2.5 | Mistral | Llama3.1 |
|---|---|---|---|---|---|---|
| **Blue 승률 (%)** | | | | | | |
| **Action Entropy (bits)** | | | | | | |
| **DCR (%)** | | | | | | |
| **TRS (1~5)** | | | | | | |
| **ESI** | | | | | | |
| **JSON 파싱률 (%)** | N/A | N/A | N/A | | | |
| **평균 추론 시간 (초/턴)** | N/A | N/A | N/A | | | |

- [ ] LLM Entropy > Rule Entropy > Script Entropy (**RQ3 가설 확인**)
- [ ] LLM DCR > 50% (**RQ1**)
- [ ] 3 LLM 모델 간 ESI/DCR 차이 존재 (**RQ2**)
- [ ] JSON 파싱 성공률 ≥ 90%
- [ ] 총 실행 게임 수:

```bash
find runs/phase4/ -name "*.jsonl" | wc -l   # 기대값: 3,000+
```

---

## Phase 5 — 통계 분석 및 논문 작성 (Week 9–12)

### 5-1. 통계 검정 실행

#### RQ1: LLM DCR > 50% — One-Sample t-test

```bash
python -c "
import json, pathlib, numpy as np
from scipy import stats

def load_dcr_scores(log_dir):
    scores = []
    for p in pathlib.Path(log_dir).rglob('*.jsonl'):
        for line in p.read_text().strip().split('\n'):
            if not line: continue
            rec = json.loads(line)
            wc = rec.get('metadata', {}).get('white_cell', {})
            dcr = wc.get('metadata', {}).get('scores', {}).get('doctrine_compliance')
            if dcr is not None:
                scores.append(float(dcr))
    return scores

for model, log_dir in [
    ('Qwen2.5-7B',   'runs/phase4/qwen/'),
    ('Mistral-7B',   'runs/phase4/mistral/'),
    ('Llama-3.1-8B', 'runs/phase4/llama/'),
]:
    dcr = load_dcr_scores(log_dir)
    if not dcr:
        print(f'{model}: 데이터 없음'); continue
    t, p = stats.ttest_1samp(dcr, popmean=0.5)
    d = (np.mean(dcr) - 0.5) / np.std(dcr, ddof=1)
    print(f'{model}: n={len(dcr)}, mean={np.mean(dcr):.3f}, t={t:.3f}, p={p:.4f}, Cohen_d={d:.3f}')
"
```

#### RQ2: 모델 간 ESI/DCR 차이 — One-Way ANOVA + Tukey HSD

```bash
python -c "
import pathlib, numpy as np
from scipy import stats
from wargame.analysis.metrics import escalation_sensitivity_index, doctrine_compliance_rate

groups = {
    'Qwen2.5-7B':   list(pathlib.Path('runs/phase4/qwen').rglob('*.jsonl')),
    'Mistral-7B':   list(pathlib.Path('runs/phase4/mistral').rglob('*.jsonl')),
    'Llama-3.1-8B': list(pathlib.Path('runs/phase4/llama').rglob('*.jsonl')),
}

print('=== ESI 비교 (ANOVA) ===')
esi_by_model = {m: [escalation_sensitivity_index(p) for p in logs] for m, logs in groups.items()}
f, p = stats.f_oneway(*esi_by_model.values())
print(f'F={f:.3f}, p={p:.4f}')
for m, vals in esi_by_model.items():
    print(f'  {m}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}')

# Tukey HSD (scipy 1.8+)
from scipy.stats import tukey_hsd
result = tukey_hsd(*esi_by_model.values())
print('Tukey HSD p-values:', result.pvalue)
"
```

#### RQ3: 에이전트 유형별 Entropy 차이 — Kruskal-Wallis + Dunn

```bash
python -c "
import pathlib, numpy as np
from scipy import stats
from scikit_posthocs import posthoc_dunn  # pip install scikit-posthocs
from wargame.analysis.metrics import action_entropy

groups = {
    'Script': list(pathlib.Path('runs/phase4/baseline/script').rglob('*.jsonl')),
    'Rule':   list(pathlib.Path('runs/phase4/baseline/rule').rglob('*.jsonl')),
    'Qwen':   list(pathlib.Path('runs/phase4/qwen').rglob('*.jsonl')),
    'Mistral':list(pathlib.Path('runs/phase4/mistral').rglob('*.jsonl')),
    'Llama':  list(pathlib.Path('runs/phase4/llama').rglob('*.jsonl')),
}
ent_by_group = {m: [action_entropy(p) for p in logs] for m, logs in groups.items()}

vals = list(ent_by_group.values())
h, p = stats.kruskal(*vals)
print(f'Kruskal-Wallis: H={h:.3f}, p={p:.4f}')
for m, v in ent_by_group.items():
    print(f'  {m}: mean={np.mean(v):.3f} bits')

# Dunn 사후 검정
import pandas as pd
all_data = [(e, m) for m, ents in ent_by_group.items() for e in ents]
df = pd.DataFrame(all_data, columns=['entropy', 'group'])
dunn = posthoc_dunn(df, val_col='entropy', group_col='group', p_adjust='bonferroni')
print('Dunn 사후 검정 (Bonferroni):\n', dunn.round(4))
"
```

#### 결과 확인

- [ ] RQ1: t-test → **p < 0.05**, Cohen's d 계산 완료
- [ ] RQ2: ANOVA + Tukey HSD 사후 검정 완료
- [ ] RQ3: Kruskal-Wallis + Dunn 사후 검정 완료
- [ ] Bonferroni 보정 적용 (α = 0.05 / 검정 수)

---

### 5-2. 시각화 생성 (matplotlib 기반 논문용 그래프)

> `plots.py`의 matplotlib 함수는 `[dev,analysis,llm-mlx,plots]` 설치 필요.
> 출력: PNG (300 DPI) + SVG 동시 저장.

```bash
mkdir -p runs/final_plots

python -c "
import pathlib, json
from wargame.analysis.metrics import (
    action_entropy, escalation_sensitivity_index, doctrine_compliance_rate
)
from wargame.analysis.plots import (
    plot_action_entropy_comparison,
    plot_win_rate_by_scenario,
    plot_esi_comparison,
    plot_dcr_distribution,
    plot_force_curve,
)

OUT = pathlib.Path('runs/final_plots')

# --- 1. Action Entropy 비교 박스플롯 (RQ3) ---
entropy_data = {}
for label, log_dir in [
    ('Script',   'runs/phase4/baseline/script'),
    ('Rule',     'runs/phase4/baseline/rule'),
    ('Qwen2.5',  'runs/phase4/qwen'),
    ('Mistral',  'runs/phase4/mistral'),
    ('Llama3.1', 'runs/phase4/llama'),
]:
    logs = list(pathlib.Path(log_dir).rglob('*.jsonl'))
    entropy_data[label] = [action_entropy(p) for p in logs]

plot_action_entropy_comparison(entropy_data, OUT / 'fig1_action_entropy')
print('Fig 1 저장 완료: fig1_action_entropy.png/.svg')

# --- 2. 시나리오별 승률 막대 그래프 ---
scenarios = ['s1_open_encounter','s2_mountain_assault','s3_urban_fight','s4_river_crossing','s5_breakout']
from wargame.analysis.metrics import win_rate
from wargame.core.enums import Faction

win_data = {}
for scenario in scenarios:
    win_data[scenario] = {}
    for label, log_dir in [('Rule','runs/phase4/baseline/rule'), ('Qwen2.5','runs/phase4/qwen')]:
        logs = list(pathlib.Path(log_dir / scenario if '/' not in log_dir else log_dir).rglob('*.jsonl'))
        if logs:
            win_data[scenario][label] = win_rate(logs, faction=Faction.BLUE)

plot_win_rate_by_scenario(win_data, OUT / 'fig2_win_rate_by_scenario')
print('Fig 2 저장 완료: fig2_win_rate_by_scenario.png/.svg')

# --- 3. ESI 비교 박스플롯 (RQ2) ---
esi_data = {}
for label, log_dir in [
    ('Qwen2.5',  'runs/phase4/qwen'),
    ('Mistral',  'runs/phase4/mistral'),
    ('Llama3.1', 'runs/phase4/llama'),
]:
    logs = list(pathlib.Path(log_dir).rglob('*.jsonl'))
    esi_data[label] = [escalation_sensitivity_index(p) for p in logs]

plot_esi_comparison(esi_data, OUT / 'fig3_esi_comparison')
print('Fig 3 저장 완료: fig3_esi_comparison.png/.svg')

# --- 4. DCR 분포 히스토그램 (RQ1) ---
dcr_data = {}
for label, log_dir in [
    ('Qwen2.5',  'runs/phase4/qwen'),
    ('Mistral',  'runs/phase4/mistral'),
    ('Llama3.1', 'runs/phase4/llama'),
]:
    logs = list(pathlib.Path(log_dir).rglob('*.jsonl'))
    dcr_data[label] = [doctrine_compliance_rate(p) for p in logs]

plot_dcr_distribution(dcr_data, OUT / 'fig4_dcr_distribution')
print('Fig 4 저장 완료: fig4_dcr_distribution.png/.svg')
"

# --- 5. 대표 게임 전투력 잔존 곡선 (별도 실행) ---
python -c "
import json, pathlib
from wargame.analysis.plots import plot_force_curve

# Qwen 대표 게임 (s1, seed 0)
log_path = next(pathlib.Path('runs/phase4/qwen/s1_open_encounter').glob('*seed_0*'), None)
if log_path:
    records = [json.loads(l) for l in log_path.read_text().strip().split('\n') if l]
    plot_force_curve(records, pathlib.Path('runs/final_plots/fig5_force_curve_qwen_s1'),
                     title='Qwen2.5-7B vs Rule — s1 (seed=0)')
    print('Fig 5 저장 완료')
"
```

#### 결과 확인

- [ ] `fig1_action_entropy.png/.svg` — 에이전트별 Action Entropy 박스플롯
- [ ] `fig2_win_rate_by_scenario.png/.svg` — 시나리오별 Blue 승률 막대 그래프
- [ ] `fig3_esi_comparison.png/.svg` — 모델별 ESI 비교 박스플롯
- [ ] `fig4_dcr_distribution.png/.svg` — DCR 분포 히스토그램
- [ ] `fig5_force_curve_*.png/.svg` — 턴별 전투력 잔존 곡선 (대표 2~3건)

---

### 5-3. 정성 사례 분석 (대표 게임 3건)

```bash
python -c "
import json, pathlib

# 대표 게임 파일 경로 지정 후 실행
log = pathlib.Path('runs/phase4/qwen/s1_open_encounter/<선택_파일>.jsonl')
records = [json.loads(l) for l in log.read_text().strip().split('\n') if l]
for rec in records:
    print(f'=== Turn {rec[\"turn\"]} ===')
    meta = rec.get('metadata', {})
    blue_r = meta.get('blue', {}).get('reasoning', '')
    red_r  = meta.get('red',  {}).get('reasoning', '')
    wc_sum = meta.get('white_cell', {}).get('reasoning', '')
    print(f'  Blue: {blue_r[:200]}')
    print(f'  Red:  {red_r[:200]}')
    print(f'  WC:   {wc_sum[:100]}')
    print()
"
```

#### 결과 확인

- [ ] 우수 의사결정 사례 1건 (교리 준수율 높음)
- [ ] 실패 사례 1건 (환각 또는 파싱 실패)
- [ ] 모델 간 전술 스타일 차이 사례 1건

---

### 5-4. 전문가 표본 검증

```bash
python -c "
import random, pathlib, shutil
all_logs = list(pathlib.Path('runs/phase4').rglob('*.jsonl'))
random.seed(42)
sample = random.sample(all_logs, max(1, len(all_logs) // 10))
out = pathlib.Path('runs/expert_sample')
out.mkdir(parents=True, exist_ok=True)
for src in sample:
    shutil.copy(src, out / src.name)
print(f'샘플 {len(sample)}개 → runs/expert_sample/')
"
```

#### 결과 확인

- [ ] 전문가-White Cell 일치도 Krippendorff's α ≥ **0.6**
- [ ] White Cell 내부 일치도(동일 로그 3회) α ≥ **0.7**

---

## 최종 검증 체크리스트

| 항목 | 기준 | 확인 |
|---|---|---|
| 총 실행 게임 수 | ≥ 3,000회 | [ ] |
| JSON 파싱 성공률 | ≥ 90% (LLM 전체) | [ ] |
| 시나리오 균형성 | Blue 승률 40~60% | [ ] |
| RQ1 t-test | p < 0.05, Cohen's d | [ ] |
| RQ2 ANOVA | p < 0.05 + Tukey HSD | [ ] |
| RQ3 Kruskal-Wallis | p < 0.05 + Dunn | [ ] |
| 전문가 일치도 | κ ≥ 0.6 | [ ] |
| 재현성 | 동일 시드 → 동일 결과 100% | [ ] |
| 로그 무결성 | 손상 파일 0개 | [ ] |
| 플롯 파일 생성 | PNG+SVG 전체 확인 | [ ] |

---

## 🗂️ 현재 진행 상황 총정리 (2026-03-21 기준)

### Phase별 완료 현황

| Phase | 항목 | 상태 | 비고 |
|---|---|---|---|
| **Phase 1** | 1-1 단위 테스트 | ✅ 완료 | 38/38 PASS |
| **Phase 1** | 1-2 단일 게임 확인 | ✅ 완료 | T04 교전 발생, Blue=292, Red=297 |
| **Phase 1** | 1-3 시나리오 5종 확인 | ✅ 완료 | s1/s3/s5 교전 ✅, s2/s4 WITHDRAW/HOLD (정상) |
| **Phase 1** | 1-4 stochastic 비교 | ⏳ 비교 미실행 | 두 jsonl 파일은 모두 생성됨 → 비교 스크립트 실행 필요 |
| **Phase 2** | 2-1 MLX 설치 | ✅ 설치 완료 | 모델 다운로드/속도 벤치마크 미완 |
| **Phase 2** | 2-2~2-4 LLM 실험 | ⏳ 대기 | 모델 다운로드 후 진행 |
| **Phase 3** | 3-0 run_batch.py 수정 | ✅ 완료 | `--visibility-radius`, `--identification-radius` 지원 |
| **Phase 3** | 3-1~3-3 베이스라인 배치 | ⏳ 대기 | 3-1 단위 테스트부터 시작 필요 |
| **Phase 4~5** | 대규모 실험/분석 | ⏳ 대기 | Phase 3 완료 후 진행 |

---

### ✅ 확인된 주요 발견 사항

1. **vr=3(기본값) → 교전 미발생**: 초기 배치 거리 6~7 hex가 감지 범위 밖 → `enemy_observations` 비어 있음
2. **idr=1(기본값) → RECON 반복**: `nearest_enemy_with_position()` 반환값이 `None`
3. **해결책 확정**: 베이스라인 `vr=8, idr=3` / LLM `vr=5, idr=2`
4. **combat 필드 키**: `combat.casualties_by_unit` (NOT `blue_loss`/`red_loss`)
5. **s2·s4 교전 없음**: Rule 에이전트가 지형 불리로 WITHDRAW/HOLD 선택 — **의도된 동작**
6. **zsh `#` 주의**: 주석 포함 블록을 한 번에 붙여넣으면 오류 → 각 명령 **개별 실행**

---

### ⚡ 지금 당장 해야 할 작업 (순서대로)

**1단계 — Phase 1 최종 마무리 (1-4 비교 스크립트)**

```bash
python -c "
import json, pathlib
for label, path in [
    ('deterministic', 'runs/phase1/s1_deterministic_vr8_idr3.jsonl'),
    ('stochastic',    'runs/phase1/s1_stochastic_vr8_idr3.jsonl'),
]:
    records = [json.loads(l) for l in pathlib.Path(path).read_text().strip().split('\n') if l]
    last = records[-1]
    units = last.get('state', {}).get('units', {})
    blue = sum(u['strength'] for u in units.values() if u.get('faction') == 'blue')
    red  = sum(u['strength'] for u in units.values() if u.get('faction') == 'red')
    combat_turns = sum(1 for r in records if r.get('combat') and r['combat'].get('casualties_by_unit'))
    print(f'{label}: Blue={blue}, Red={red}, 교전={combat_turns}턴')
"
```

기대 결과: 결정론적 vs 확률적 최종 잔존 병력이 **다름** (stochastic noise 효과)

---

**2단계 — Phase 3-1 베이스라인 단위 테스트**

```bash
python -m pytest tests/test_baseline_agents.py -v
```

---

**3단계 — Phase 2-1 Qwen 모델 다운로드 및 속도 확인 (Mac M4)**

```bash
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-7B-Instruct-4bit')"
```

> 모델 크기 약 4~5GB — 다운로드 시간 감안하여 백그라운드에서 진행 가능

---

### ⚠️ 검토가 필요한 사항

| 항목 | 내용 | 우선도 |
|---|---|---|
| `lanchester.py` docstring 오류 | "square-law" 표기이나 실제 linear law 구현 | 낮음 (논문 기재 시 주의) |
| `_allocate_losses` O(n²) | 유닛 수 많을 때 성능 저하 가능 | 낮음 (현 규모에서 무관) |
| Python ≥ 3.11 필요 | `StrEnum` 사용 — 3.10에서 실행 불가 | **확인 완료** (Mac 3.13.5 ✅) |
| s2·s4 LLM 교전 여부 | Rule은 WITHDRAW/HOLD이나 LLM은 공격적일 수 있음 | Phase 4에서 관찰 예정 |
| `run_batch.py` `--visibility-radius` 기본값 | 현재 기본값 3 → 실행 시 반드시 `--visibility-radius 8` 명시 | **실행 시 주의** |

---

## 빠른 참조 — 주요 명령어 모음

```bash
# 의존성 설치
pip install -e ".[dev,analysis,llm-mlx]"          # Mac M4 기본
pip install -e ".[dev,analysis,llm-mlx,plots]"    # 시각화 포함
pip install -e ".[dev,analysis]" && pip install vllm  # Colab A100

# 단일 게임 실행
python scripts/run_single_game.py \
  --scenario <SCENARIO_ID> \
  --blue-agent <AGENT_SPEC> \
  --red-agent  <AGENT_SPEC> \
  --visibility-radius <8|5> \      # 베이스라인=8, LLM=5
  --identification-radius <3|2> \  # 베이스라인=3, LLM=2
  --seed <N> \
  --stochastic-combat --noise-std 0.1 \
  --output <path>.jsonl

# 배치 실험 실행
python scripts/run_batch.py \
  --scenario <SCENARIO_ID> \       # 반복 가능
  --matchup "<blue_spec>,<red_spec>" \
  --seed-count <N> \
  --visibility-radius <8|5> \
  --identification-radius <3|2> \
  --backend <auto|mlx|vllm> \
  --stochastic-combat \
  --noise-std 0.1 \
  --output-dir <dir>

# 로그 분석 및 지표 출력
python scripts/evaluate_logs.py <dir_or_files> \
  --plot-dir <plot_dir>

# 테스트
python -m pytest tests/ -v
python -m pytest tests/test_lanchester.py -v
```

### 에이전트 스펙 (`AGENT_SPEC`) 표기

| 표기 | 의미 | 환경 |
|---|---|---|
| `rule` | Rule-Based Agent | 모든 환경 |
| `random` | Random Agent | 모든 환경 |
| `script` | Script Agent (기본: frontal_assault) | 모든 환경 |
| `script:frontal_assault` | 정면공격 스크립트 | 모든 환경 |
| `script:flank_maneuver` | 우회기동 스크립트 | 모든 환경 |
| `script:delay_defense` | 지연전 스크립트 | 모든 환경 |
| `local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit` | Qwen2.5-7B (MLX 4bit) | Mac M4 |
| `local_llm:mistralai/Mistral-7B-Instruct-v0.3` + `--backend vllm` | Mistral-7B (vLLM) | Colab A100 |
| `local_llm:meta-llama/Llama-3.1-8B-Instruct` + `--backend vllm` | Llama-3.1-8B (vLLM) | Colab A100 |

### 시나리오 ID 목록

| ID | 시나리오명 | 핵심 전술 |
|---|---|---|
| `s1_open_encounter` | 평지 조우전 | 기동의 자유, 측면 기동 |
| `s2_mountain_assault` | 산악 방어진지 공격 | 지형 활용, 공격 경로 선택 |
| `s3_urban_fight` | 시가지 전투 | 근접 전투, 건물 활용 |
| `s4_river_crossing` | 하천 도하 작전 | 취약 시점 전투력 집중 |
| `s5_breakout` | 포위 돌파 | 비대칭 상황, 위기 의사결정 |

### Fog-of-War 파라미터 규칙 요약 (2차 수정 확정)

| 실험 유형 | `--visibility-radius` | `--identification-radius` | 근거 |
|---|---|---|---|
| 베이스라인 (rule/random/script) | **8** | **3** | 초기 배치 거리(6~7) 커버, 교전 보장 ✅ |
| LLM 에이전트 안정성 테스트 | **5** | **2** | 중간 불확실성 유지 |
| LLM 에이전트 대규모 실험 | **5** | **2** | 중간 불확실성 유지 |

> **검증 완료**: s1 vr=8/idr=3 → T04 교전 발생 (Blue -8, Red -3) ✅
> s2·s4 vr=8/idr=3 → Rule 에이전트 WITHDRAW/HOLD (정상 동작) ✅

---

*최종 수정: 2026-03-22 (todo_list 최종 정리, 빠른 참조 vr 파라미터 수정, s2·s4 vr=8 결과 확정 기록)*
