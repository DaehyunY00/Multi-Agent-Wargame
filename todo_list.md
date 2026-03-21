# 실험 진행 체크리스트 — LLM 기반 다중 에이전트 워게임

> **진행 규칙**
> - 각 단계 완료 시 `[ ]` → `[x]`로 변경
> - `# 결과 확인` 블록의 기준을 **통과해야만** 다음 단계로 진행
> - 실험 로그는 모두 `runs/` 하위에 저장, `.gitignore`에서 관리
> - 컴퓨팅 환경: **Mac M4** (Python 3.13, mlx-lm) / **Colab Pro A100** (vLLM)

> **⚠️ 중요 파라미터 규칙 — Phase 1 검증에서 발견**
>
> | 실험 목적 | `identification_radius` 설정 방법 | 이유 |
> |---|---|---|
> | `run_single_game.py` 테스트 | `--identification-radius 3` CLI 인자 사용 | 인자 지원됨 |
> | `run_batch.py` 배치 실험 | **코드 수정 필요** (아래 Phase 3-0 참조) | CLI 인자 미구현 |
>
> **배경**: `identification_radius=1(기본값)`이면 Rule 에이전트가 거리 2~3의 적을 감지만 하고
> 위치를 알 수 없어 `RECON` 행동만 반복 → 교전 미발생. Phase 1에서 s1·s2·s4가
> 교전 0회, 병력 손실 0으로 종료된 원인.
>
> **추가 발견**: `run_batch.py`는 `FogOfWarFilter()`를 인자 없이 호출하여
> `identification_radius=1`이 항상 고정됨. `--identification-radius` CLI 인자 자체가
> 존재하지 않아 Phase 3 배치 실험 전에 코드 수정이 필요함.

---

## 환경 설정

```bash
# 최초 1회만 실행
cd Multi-Agent_Wargame
pip install -e ".[dev]"
mkdir -p logs runs

# 의존성 추가 설치 (실험 전 전체 설치)
pip install numpy scipy matplotlib seaborn pandas PyYAML mlx-lm
```

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

### 1-2. 단일 게임 동작 확인 (Rule vs Rule, identification-radius 3 적용)

> ⚠️ `--identification-radius 3` 추가 필수 — 미적용 시 s1에서 교전이 발생하지 않음

```bash
mkdir -p runs/phase1

python scripts/run_single_game.py \
  --scenario s1_open_encounter \
  --blue-agent rule \
  --red-agent rule \
  --identification-radius 3 \
  --output runs/phase1/s1_rule_vs_rule_idr3.jsonl
```

#### 결과 확인

- [x] 종료 메시지에 `"terminal": true` 포함 ✅
- [x] `runs/phase1/s1_rule_vs_rule.jsonl` 파일 생성 ✅ (기존 실행 확인)
- [ ] `"turns"` 값이 8~20 범위 내 (idr3 적용 후 재확인)
- [ ] 교전 발생 확인 (`교전 발생 턴 수 > 0`)

> **기존 실행 결과 (identification-radius=1 기본값)**
> - s1: turns=12 (=max), 교전 0회, 병력 손실 0 → **교전 미발생** ⚠️
> - idr3 재실행으로 교전 발생 여부 확인 필요

---

### 1-3. 시나리오 5종 × Rule 에이전트 동작 확인 (identification-radius 3 적용)

```bash
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_single_game.py \
    --scenario $SCENARIO \
    --blue-agent rule \
    --red-agent rule \
    --identification-radius 3 \
    --output runs/phase1/${SCENARIO}_rule_idr3.jsonl
  echo "Done: $SCENARIO"
done
```

#### 결과 확인 (기존 기본값 실행 결과 기록)

| 시나리오 | 기존(idr=1) 교전 | 기존 결과 | idr=3 교전 | idr=3 결과 |
|---|---|---|---|---|
| s1_open_encounter | 0/12턴 ⚠️ | DRAW (손실 0) | | |
| s2_mountain_assault | 0/14턴 ⚠️ | DRAW (손실 0) | | |
| s3_urban_fight | 10/12턴 ✅ | RED 승 (B:228, R:235) | | |
| s4_river_crossing | 0/14턴 ⚠️ | DRAW (손실 0) | | |
| s5_breakout | 5/13턴 ✅ | BLUE 승 (B:278, R:253) | | |

- [ ] idr=3 적용 후 s1·s2·s4 **교전 발생 확인** (교전 턴 > 0)
- [x] s3, s5는 기존 기본값에서도 교전 발생 ✅
- [x] 모든 jsonl 파일 크기 > 0 ✅ (기존 7개 파일 확인)
- [x] 모든 게임 `terminal: true` ✅

---

### 1-4. Lanchester 확률적 모드 확인 (identification-radius 3 적용)

> idr=1 기본값 실행 시 교전이 없어 stochastic/deterministic 결과가 동일했음.
> idr=3으로 교전이 발생해야 확률적 노이즈 효과를 확인할 수 있음.

```bash
# 결정론적 (기준)
python scripts/run_single_game.py \
  --scenario s1_open_encounter \
  --blue-agent rule --red-agent rule \
  --identification-radius 3 \
  --output runs/phase1/s1_deterministic_idr3.jsonl

# 확률적
python scripts/run_single_game.py \
  --scenario s1_open_encounter \
  --blue-agent rule --red-agent rule \
  --identification-radius 3 \
  --stochastic-combat --noise-std 0.1 \
  --output runs/phase1/s1_stochastic_idr3.jsonl
```

#### 결과 확인

- [ ] 두 실행의 최종 잔존 병력이 **다름** (stochastic 노이즈 효과 확인)
- [ ] 오류 없이 종료

---

## Phase 2 — LLM 에이전트 시스템 구축 (Week 3–4)

### 2-1. MLX 환경 구축 (Mac M4)

```bash
# mlx-lm 설치 및 모델 다운로드
pip install mlx-lm

# Qwen2.5-7B-Instruct 4bit 양자화 모델 다운로드
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-7B-Instruct-4bit')"
```

#### 결과 확인

- [ ] `mlx-lm` import 오류 없음
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

### 2-2. JSON 출력 안정성 검증 (수동 10회 테스트)

> LLM 에이전트 실험에는 `--identification-radius 2` 적용 (불확실성 유지 + 교전 발생 균형)

```bash
mkdir -p runs/phase2/llm_stability

python scripts/run_single_game.py \
  --scenario s1_open_encounter \
  --blue-agent local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit \
  --red-agent rule \
  --identification-radius 2 \
  --output runs/phase2/llm_stability/qwen_test_01.jsonl

python scripts/evaluate_logs.py runs/phase2/llm_stability/
```

#### 결과 확인

- [ ] JSON 파싱 성공률 `"json_parsing_success_rate"` ≥ **0.90**
- [ ] `"used_fallback"` 비율 < 10%
- [ ] 비정상 종료 없음

---

### 2-3. 프롬프트 반복 개선 (10회 수동 테스트)

```bash
for i in $(seq 1 10); do
  python scripts/run_single_game.py \
    --scenario s2_mountain_assault \
    --blue-agent local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit \
    --red-agent rule \
    --identification-radius 2 \
    --seed $i \
    --output runs/phase2/llm_stability/prompt_test_s${i}.jsonl
done

python scripts/evaluate_logs.py runs/phase2/llm_stability/ \
  --plot-dir runs/phase2/plots/
```

#### 결과 확인

- [ ] 10회 중 오류 종료 **0회**
- [ ] `"mean_action_entropy"` > 0 (결정론적이지 않음)
- [ ] `"mean_doctrine_compliance_rate"` > 0.3
- [ ] `plots/` 하위에 SVG 파일 생성 확인

---

### 2-4. Colab 환경 구성 (Mistral-7B, Llama-3.1-8B)

> Colab에서 실행 — 아래 명령어를 Colab 노트북 셀에 붙여넣기

```bash
!pip install vllm bitsandbytes transformers accelerate
!git clone https://github.com/<your_repo>/Multi-Agent_Wargame.git
%cd Multi-Agent_Wargame
!pip install -e ".[dev]"

!python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.3', quantization='bitsandbytes', load_format='bitsandbytes')
print('Mistral 로드 완료')
"
```

#### 결과 확인

- [ ] Colab A100에서 vLLM 설치 완료
- [ ] Mistral-7B 로드 성공, 추론 속도 **70 tok/s 이상**
- [ ] Llama-3.1-8B 로드 성공, 추론 속도 **70 tok/s 이상**

---

## Phase 3 — 베이스라인 시스템 검증 (Week 5)

### 3-0. ⚠️ 사전 필수 코드 수정 — `run_batch.py` identification-radius 지원 추가

> `run_batch.py`의 `_build_runner` 함수가 `FogOfWarFilter()`를 인자 없이 호출하여
> `identification_radius`가 항상 1로 고정됩니다. Phase 3 배치 실험 전에 반드시 수정하세요.

**수정 위치**: `scripts/run_batch.py`

```python
# 1) argparse에 인자 추가 (main 함수 parser 블록 안)
parser.add_argument("--visibility-radius",    type=int, default=3)
parser.add_argument("--identification-radius", type=int, default=1)

# 2) _build_runner 함수 시그니처에 파라미터 추가
def _build_runner(*, condition, white_cell_spec, stochastic_combat, noise_std,
                  visibility_radius=3, identification_radius=1):

# 3) _build_runner 내부 FogOfWarFilter 호출 수정
fog = FogOfWarFilter(
    visibility_radius=visibility_radius,
    identification_radius=identification_radius,
)

# 4) batch.run()을 호출하는 runner_factory 람다에서 파라미터 전달
runner_factory=lambda condition: _build_runner(
    condition=condition,
    white_cell_spec=args.white_cell,
    stochastic_combat=args.stochastic_combat,
    noise_std=args.noise_std,
    visibility_radius=args.visibility_radius,
    identification_radius=args.identification_radius,
),
```

**수정 확인**:

```bash
# 수정 후 단순 동작 확인 (소규모 배치)
python scripts/run_batch.py \
  --scenario s1_open_encounter \
  --matchup rule,rule \
  --seed-count 3 \
  --identification-radius 3 \
  --output-dir runs/test_idr3/

python scripts/evaluate_logs.py runs/test_idr3/
```

- [ ] `--identification-radius` 인자 추가 및 코드 수정 완료
- [ ] 소규모 배치(3회) 실행 후 교전 발생 확인 (`mean_action_entropy` > 0)

---

### 3-1. 베이스라인 단위 테스트

```bash
python -m pytest tests/test_baseline_agents.py -v
```

#### 결과 확인

- [ ] 전체 통과

---

### 3-2. 베이스라인 3종 × 시나리오 5종 × 50회 반복

> ⚠️ **Phase 3-0의 코드 수정 완료 후 실행** — `--identification-radius` 미수정 시 에러 발생
>
> 베이스라인: `--identification-radius 3` / LLM 실험: `--identification-radius 2`

```bash
mkdir -p runs/phase3/{rule,random,script}

# Rule-Based: 5 시나리오 × 50 seed
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup rule,rule \
    --seed-count 50 \
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
    --identification-radius 3 \
    --output-dir runs/phase3/script/${SCENARIO}
  echo "[Script] Done: $SCENARIO"
done
```

#### 결과 확인 (시나리오 균형성)

| 시나리오 | Rule 승률(Blue) | Random Entropy | Script Entropy |
|---|---|---|---|
| s1_open_encounter | ___% | ___ bits | ___ bits |
| s2_mountain_assault | ___% | ___ bits | ___ bits |
| s3_urban_fight | ___% | ___ bits | ___ bits |
| s4_river_crossing | ___% | ___ bits | ___ bits |
| s5_breakout | ___% | ___ bits | ___ bits |

- [ ] 모든 시나리오 Blue 승률 **40~60%** 범위 내
- [ ] 교착(평균 게임 길이 = max_turns) 빈도 < 20%
- [ ] Script 에이전트 엔트로피 ≈ 0 (완전 결정론적)

```bash
# 시나리오별 통계 출력
for AGENT in rule random script; do
  echo "=== $AGENT ==="
  python scripts/evaluate_logs.py runs/phase3/${AGENT}/
done
```

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
- [ ] `plots/` 하위 시각화 파일 확인

---

## Phase 4 — 대규모 실험 (Week 6–8)

> **주의**: LLM 에이전트 실험은 Colab A100에서 실행.
> 베이스라인 비교 실험은 Mac M4에서 병렬 실행 가능.
>
> **파라미터 구분**:
> - 베이스라인: `--identification-radius 3` (교전 보장)
> - LLM 에이전트: `--identification-radius 2` (불확실성 유지 + 교전 발생)

### 4-1. LLM 실험: Qwen2.5-7B × 5 시나리오 × 100회 (Mac M4)

```bash
mkdir -p runs/phase4/qwen

for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "local_llm:mlx-community/Qwen2.5-7B-Instruct-4bit,rule" \
    --seed-count 100 \
    --stochastic-combat \
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
- [ ] 중간 로그 무결성 확인:

```bash
find runs/phase4/qwen/ -name "*.jsonl" | wc -l   # 기대값: 500

python -c "
import json, pathlib
errors = []
for p in pathlib.Path('runs/phase4/qwen').rglob('*.jsonl'):
    try:
        [json.loads(l) for l in p.read_text().strip().split('\n') if l]
    except Exception as e:
        errors.append((str(p), str(e)))
print(f'손상 파일: {len(errors)}')
"
```

---

### 4-2. LLM 실험: Mistral-7B × 5 시나리오 × 100회 (Colab A100)

```bash
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "local_llm:mistralai/Mistral-7B-Instruct-v0.3,rule" \
    --seed-count 100 \
    --stochastic-combat \
    --identification-radius 2 \
    --output-dir /content/drive/MyDrive/wargame_runs/mistral/${SCENARIO}
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
for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
  python scripts/run_batch.py \
    --scenario $SCENARIO \
    --matchup "local_llm:meta-llama/Llama-3.1-8B-Instruct,rule" \
    --seed-count 100 \
    --stochastic-combat \
    --identification-radius 2 \
    --output-dir /content/drive/MyDrive/wargame_runs/llama/${SCENARIO}
done
```

#### 결과 확인

- [ ] 5개 시나리오 모두 완료
- [ ] JSON 파싱 성공률 ≥ **90%**

---

### 4-4. 베이스라인 비교 실험: 100회 반복 (Mac M4)

> Phase 3의 50회를 100회로 확장. 이미 완료된 경우 스킵 가능.

```bash
mkdir -p runs/phase4/baseline/{rule,random,script}

for AGENT_PAIR in "rule,rule" "random,random" "script:frontal_assault,script:frontal_assault"; do
  LABEL=$(echo $AGENT_PAIR | tr ':' '-' | tr ',' '_vs_')
  for SCENARIO in s1_open_encounter s2_mountain_assault s3_urban_fight s4_river_crossing s5_breakout; do
    python scripts/run_batch.py \
      --scenario $SCENARIO \
      --matchup "$AGENT_PAIR" \
      --seed-count 100 \
      --identification-radius 3 \
      --output-dir runs/phase4/baseline/${LABEL}/${SCENARIO}
  done
done
```

---

### 4-5. White Cell 배치 평가

```bash
python scripts/evaluate_logs.py runs/phase4/qwen/   --plot-dir runs/phase4/plots/qwen/
python scripts/evaluate_logs.py runs/phase4/mistral/ --plot-dir runs/phase4/plots/mistral/
python scripts/evaluate_logs.py runs/phase4/llama/   --plot-dir runs/phase4/plots/llama/
python scripts/evaluate_logs.py runs/phase4/baseline/ --plot-dir runs/phase4/plots/baseline/
```

---

### 4-6. 전체 결과 취합 및 논문 지표 기록

```bash
python scripts/evaluate_logs.py \
  runs/phase4/qwen/ runs/phase4/mistral/ runs/phase4/llama/ runs/phase4/baseline/ \
  > runs/phase4/summary_all.json

cat runs/phase4/summary_all.json
```

#### 결과 확인 — 논문 핵심 지표 기록

| 지표 | Script | Rule | Random | Qwen2.5 | Mistral | Llama3.1 |
|---|---|---|---|---|---|---|
| **Blue 승률 (%)** | | | | | | |
| **Action Entropy (bits)** | | | | | | |
| **DCR (%)** | | | | | | |
| **TRS (1~5)** | | | | | | |
| **ESI** | | | | | | |
| **TTF** | | | | | | |
| **JSON 파싱률 (%)** | N/A | N/A | N/A | | | |
| **평균 추론 시간 (초/턴)** | N/A | N/A | N/A | | | |

- [ ] LLM Entropy > Rule Entropy > Script Entropy (**RQ3 가설 확인**)
- [ ] LLM DCR > 50% (t-test 유의, **RQ1**)
- [ ] 3 LLM 모델 간 ESI/DCR 차이 존재 (**RQ2**)
- [ ] JSON 파싱 성공률 ≥ 90% (모든 LLM)
- [ ] 총 실행 게임 수:

```bash
find runs/phase4/ -name "*.jsonl" | wc -l   # 기대값: 3,000+
```

---

## Phase 5 — 통계 분석 및 논문 작성 (Week 9–12)

### 5-1. 통계 검정 실행

```bash
# RQ1: one-sample t-test (DCR > 50%)
python -c "
from scipy import stats
# dcr_values = [...]   # evaluate_logs 결과에서 수집
# t_stat, p_val = stats.ttest_1samp(dcr_values, popmean=0.5)
# print(f't={t_stat:.3f}, p={p_val:.4f}')
print('분석 스크립트 작성 필요')
"
# RQ2: one-way ANOVA  (Qwen vs Mistral vs Llama — ESI, DCR)
# RQ3: Kruskal-Wallis (Script vs Rule vs LLM — Action Entropy)
```

#### 결과 확인

- [ ] RQ1: t-test → **p < 0.05**
- [ ] RQ2: ANOVA + Tukey HSD 사후 검정 완료
- [ ] RQ3: Kruskal-Wallis + Dunn 사후 검정 완료
- [ ] Cohen's d 효과 크기 계산 완료
- [ ] Bonferroni 보정 적용 (α = 0.05 / 검정 수)

---

### 5-2. 시각화 생성

```bash
python scripts/evaluate_logs.py runs/phase4/ --plot-dir runs/final_plots/
```

#### 결과 확인

- [ ] 에이전트별 Action Entropy 박스플롯
- [ ] 시나리오별 Blue 승률 막대 그래프
- [ ] 모델별 ESI 비교 차트
- [ ] 턴별 전투력 잔존 곡선 (대표 게임 2~3건)

---

### 5-3. 정성 사례 분석 (대표 게임 3건)

```bash
python -c "
import json, pathlib

log = pathlib.Path('runs/phase4/qwen/s1_open_encounter/<선택_파일>.jsonl')
for line in log.read_text().strip().split('\n'):
    record = json.loads(line)
    print(f'=== Turn {record[\"turn\"]} ===')
    meta = record.get('metadata', {})
    print('  Blue:', meta.get('blue', {}).get('reasoning', '')[:200])
    print('  Red: ', meta.get('red',  {}).get('reasoning', '')[:200])
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
| RQ1 t-test | p < 0.05 | [ ] |
| RQ2 ANOVA | p < 0.05 + Tukey HSD | [ ] |
| RQ3 Kruskal-Wallis | p < 0.05 + Dunn | [ ] |
| 전문가 일치도 | κ ≥ 0.6 | [ ] |
| 재현성 | 동일 시드 → 동일 결과 100% | [ ] |
| 로그 무결성 | 손상 파일 0개 | [ ] |
| 플롯 파일 생성 | SVG 전체 확인 | [ ] |

---

## 빠른 참조 — 주요 명령어 모음

```bash
# 단일 게임 실행
python scripts/run_single_game.py \
  --scenario <ID> \
  --blue-agent <rule|random|script|script:BEHAVIOR|local_llm:MODEL> \
  --red-agent  <rule|random|script|script:BEHAVIOR|local_llm:MODEL> \
  --identification-radius <1|2|3> \   # 베이스라인=3, LLM=2
  --seed <N> \
  --output <path>.jsonl

# 배치 실험 실행
python scripts/run_batch.py \
  --scenario <ID> \              # 반복 가능
  --matchup "<blue>,<red>" \     # 반복 가능
  --seed-count <N> \
  --identification-radius <3|2> \
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

### 시나리오 ID 목록

| ID | 시나리오명 | 핵심 전술 |
|---|---|---|
| `s1_open_encounter` | 평지 조우전 | 기동의 자유, 측면 기동 |
| `s2_mountain_assault` | 산악 방어진지 공격 | 지형 활용, 공격 경로 선택 |
| `s3_urban_fight` | 시가지 전투 | 근접 전투, 건물 활용 |
| `s4_river_crossing` | 하천 도하 작전 | 취약 시점 전투력 집중 |
| `s5_breakout` | 포위 돌파 | 비대칭 상황, 위기 의사결정 |

### 에이전트 스펙 표기

| 표기 | 의미 |
|---|---|
| `rule` | Rule-Based Agent |
| `random` | Random Agent |
| `script` | Script Agent (기본: frontal_assault) |
| `script:frontal_assault` | 정면공격 스크립트 |
| `script:flanking_maneuver` | 우회기동 스크립트 |
| `script:delay` | 지연전 스크립트 |
| `local_llm:<HF_MODEL_ID>` | 로컬 LLM 에이전트 (MLX / vLLM) |

---

*최종 수정: 2026-03-21 (Phase 1 검증 반영 — identification-radius 수정)*
