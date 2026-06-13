# Reducing Initial KV Cache (RIC)

**Streaming KV-cache compression that bounds the *prefill peak* by interleaving reconstruction-based eviction with the prefill itself.**

KVzip / SnapKV / PyramidKV 등 기존 prefill 압축 기법은 **긴 context의 KV를 전부 만든 뒤(materialize) 압축**한다. 그래서 압축률을 낮춰도 *peak* 메모리는 full KV 그대로이고, 같은 GPU에서 받을 수 있는 초기 context 길이의 상한은 baseline과 같다(오히려 KVzip은 scoring forward 때문에 peak가 약간 더 높다). 본 레포(`paper.tex`의 방향)는 **prefill 도중 압축을 interleave**해서 물리적으로 저장되는 KV가 항상 target budget 안에 머물도록 하고, 그 결과 prefill peak를 budget이 정한다(고정 budget이면 $L$ 무관, ratio budget이면 $O(rL)$). 같은 메커니즘을 multi-turn에도 적용해 turn 누적도 ratio $r$로 throttling한다.

> 상태: **구현·실험 완료.** NIAH grid (qwen3-4b / llama-3.2-3b × ctx 500/2k/8k × ratio 0~1 × 6 method) + SCBench multi-turn (summary / kv)까지 모두 돌렸고, 결과는 `results/`와 `RESULTS_NOTES.md`에 정리되어 있으며 `paper.tex`의 표/숫자에 반영되어 있다.

---

## 1. 핵심 아이디어

물리 레이아웃: `[sink | committed (frozen) | working]`. context를 chunk 단위로 나누어 다음을 반복한다.

```
for chunk in context_chunks:
    prefill(chunk)                              # dense append -> working window
    score(chunk)                                # KVzip reconstruction ("repeat") on the fresh chunk
    light_drop(chunk)                           # chunk score < ratio*chunk_max 인 토큰 경량 삭제
    if working_len >= working_max:              # heavy-prune trigger
        rescore(working) [optional]             # working window만 repeat 프롬프트로 재스코어
        prune working -> top intermediate_budget B_i  # commit & freeze, working 리셋
        if committed_len > final_budget B_f:    # 누적 캡
            reprune committed -> top B_f
finalize: per_token (dense) or per_head (KVzip varlen) eviction
```

- **왜 peak가 낮은가:** 물리 저장 KV가 항상 `sink + B_f + working_max + chunk` 수준으로 bound. 누적 context 길이와 무관. baseline은 full KV를 만든 뒤 압축하므로 같은 길이에서 KVzip이 OOM이어도 ours는 budget 내에서 통과.
- **두 단계(Hybrid):** chunk마다 절대 threshold로 *경량* 삭제(`light_drop_ratio`) + working window가 trigger에 도달하면 hard top-K(`intermediate_budget`)로 *강하게* 추려 commit. 누적(committed)은 `final_budget`(=B_i의 정수배)에서만 importance 재prune.
- **두 가지 압축 단위 (`--level`):** `per_token`(dense, 멀티턴 append 가능) / `per_head`(KVzip 방식 varlen, 최종 finalize 시 적용 → answer 단계 추가 절감).

### 중요도 신호 변종 (`paper.tex` ablation §4.4)
- `repeat` (기본, KVzip 재구성): "repeat the chunk" 프롬프트로 점수 — 가장 안정적.
- `predict-self` v1 ("Predict the entire context:") / v2 ("Predict the upcoming context:") — `--use_predict_prompt --predict_prompt_version {1,2}`.
- `predict-next` (실제 다음 chunk를 query로 점수, sink/recent 보호) — `--predict_target next_chunk`.
- `combine` (per-position max of repeat & predict) — `--combine_repeat`.
- 모든 변종에서 **최종 compaction은 항상 repeat**: 답변용 cache는 항상 reconstruction-faithful.

### Multi-turn 확장 (`paper.tex` §3.3)
초기 context를 ratio $r$로 압축했으면 각 turn(query+answer)도 **그 turn 길이 $L_t$의 `round(turn_ratio · L_t)`**로 동적 압축. 세션 전체가 같은 비율로 균일 압축됨 → turn 누적 ≈ $r \times$ baseline. legacy "fixed cap" 모드(`turn_ratio < 0`)도 호환 유지.

### Gappy-RoPE 근사
prune된 key의 RoPE 위치는 원본 그대로 유지하고 재배열하지 않음 → 물리 배열 순서 = logical 순서 → flash causal attention 그대로 성립. KVzip이 post-prune inference에서 이미 의존하는 근사를 **prefill 도중**으로 한 단계 앞당겨 적용.

---

## 2. 코드 구조

```
Reducing-Initial-KVcache/
├── ric/
│   ├── __init__.py        # KVzip을 sys.path에 주입 (설치 없이 import)
│   ├── stream_cache.py    # StreamingCache (EvictCache 상속): [sink|committed|working] 레이아웃,
│   │                      # light_drop, prune_working_to_segment, reprune_committed_to_final,
│   │                      # reprune_turns_to_cap (legacy multi-turn cap), finalize per_token/per_head.
│   ├── stream_prefill.py  # RICConfig + streaming_prefill (chunk loop, scoring, prune trigger) +
│   │                      # multiturn_compress (dynamic turn_ratio + legacy fixed-cap) +
│   │                      # derive_ric_budgets (comp_ratio -> B_f/B_i/working_max).
│   ├── baselines.py       # full / kvzip / streamllm / ours 디스패치.
│   ├── mem.py             # reset_peak / peak_gb / run_oom_safe (torch.cuda.max_memory_allocated).
│   └── record.py          # capture_env / save_run (실행 JSON + jsonl 누적).
├── args.py                # 공용 CLI (모든 RICConfig 필드 + comp_ratio + predict/combine 변종).
├── run_niah.py            # 단일 NIAH 셀 (ctx 사다리/길이 sweep).
├── run_niah_grid.py       # paper §4.2: NIAH per-head 그리드 (ctx × ratio × method).
├── run_scbench.py         # paper §4.3: SCBench multi-turn (summary / kv).
├── aggregate.py           # results/*.jsonl -> summary_*.md (paper 표 자동 생성).
├── RESULTS_NOTES.md       # 전체 실험 staging 한국어 분석 (NIAH + Prompt study + SCBench).
├── SETUP.md               # 새 서버에서 처음 jk env 만들 때의 단계별 함정 정리.
└── results/
    ├── niah_grid/         # 셀별 JSON
    ├── scbench/           # 실행별 JSON
    ├── niah_grid_runs.jsonl, scbench_runs.jsonl  # flat 누적 요약
    └── summary_niah_grid.md, summary_scbench.md  # aggregate 결과 (paper 표 원천)
```

KVzip은 형제 디렉토리 `../KVzip/`에 있어야 한다(설치 X — KVzip의 pyproject가 torch/numpy를 다운그레이드시키므로 일부러 import-only). 위치가 다르면 `KVZIP_DIR` 환경변수.

---

## 3. 환경 세팅

이 레포는 KVzip을 런타임 의존성으로 import한다. 본 실험은 **L4 24GB GPU + Ubuntu + conda env `jk`**에서 돌렸고, 환경 구축 자체는 [`SETUP.md`](SETUP.md)에 단계별로 정리되어 있다. 핵심 요약:

- Python 3.12 + `requirements.txt`
- 서버 CUDA에 맞는 PyTorch (본 실험은 torch 2.8.0 + CUDA 12.9)
- **transformers는 4.51.3 고정** (KVzip이 의존하는 attention API 시그니처에 매우 민감)
- flash-attn 2.8.3 (서버 GPU arch로 직접 빌드)
- KVzip은 **`pip install -e .` 금지** — `../KVzip/`에 clone만 하고, `KVzip/csrc/build.py`의 GPU arch를 서버에 맞게 수정 후 커널 빌드
- `KVZIP_DIR` 환경변수로 KVzip 경로 지정 가능 (기본: 형제 디렉토리)

freeze된 패키지 목록은 `env/jk-conda-full.yml`, `env/jk-pip-freeze.txt`.

이후 모든 실행은 `conda run -n jk ...` 또는 `conda activate jk` 후 진행.

---

## 4. 재현 — 실험 1 (NIAH grid: prefill peak + 정확도)

paper §4.2. 작은 fixed context(500 / 2000 / 8000 tokens)에서 (ratio × method)의 prefill peak 메모리와 needle 정확도(11 quantile depth 평균)를 동시에 측정. 같은 (ctx_len, depth)의 token ids는 한 번 생성되어 모든 셀에 동일 입력.

### Qwen3-4B (primary)
```bash
conda run -n jk python run_niah_grid.py -m qwen3-4b \
    --ctx_lens 500,2000,8000 \
    --depths 0,10,20,30,40,50,60,70,80,90,100 \
    --ratios 0.0,0.2,0.4,0.6,0.8,1.0 \
    --methods kvzip,ours,ours_predict_v1,ours_predict_v2,ours_predict_next,ours_combine
```

### Llama-3.2-3B
```bash
conda run -n jk python run_niah_grid.py -m llama3.2-3b \
    --ctx_lens 500,2000,8000 \
    --depths 0,10,20,30,40,50,60,70,80,90,100 \
    --ratios 0.0,0.2,0.4,0.6,0.8,1.0 \
    --methods kvzip,ours,ours_predict_v1,ours_predict_v2,ours_predict_next,ours_combine
```

각 셀의 raw record는 `results/niah_grid/<model>_<method>_<level>_<timestamp>.json`, flat 요약은 `results/niah_grid_runs.jsonl`. paper §4.2의 Table 1·2·3 모두 이 그리드 출력에서 직접 발췌.

---

## 5. 재현 — 실험 2 (SCBench multi-turn)

paper §4.3. 공유 long context를 한 번 prefill한 뒤, 질문을 순환하며 multi-turn 대화를 누적. ours는 각 (query+answer) turn을 즉시 `round(turn_ratio · L_t)`로 동적 압축; kvzip은 context만 압축하고 turn은 무한 누적 (baseline).

### summary (compressible NL, 약 10k context, 100 turns, context r=0.4)
```bash
# ours (repeat 기본)
conda run -n jk python run_scbench.py -m qwen3-4b -d scbench_summary_short \
    --method ours --comp_ratio 0.4 --turn_ratio 0.4 \
    --turns 100 --consec_zero_stop 5

# kvzip baseline (동일 retained ratio)
conda run -n jk python run_scbench.py -m qwen3-4b -d scbench_summary_short \
    --method kvzip --ratio 0.4 --turns 100 --consec_zero_stop 5

# 프롬프트 변종 (prompt study)
conda run -n jk python run_scbench.py -m qwen3-4b -d scbench_summary_short \
    --method ours --comp_ratio 0.4 --turn_ratio 0.4 \
    --use_predict_prompt --predict_prompt_version 1 --turns 100
conda run -n jk python run_scbench.py -m qwen3-4b -d scbench_summary_short \
    --method ours --comp_ratio 0.4 --turn_ratio 0.4 \
    --use_predict_prompt --predict_prompt_version 2 --turns 100
conda run -n jk python run_scbench.py -m qwen3-4b -d scbench_summary_short \
    --method ours --comp_ratio 0.4 --turn_ratio 0.4 \
    --combine_repeat --turns 100
```

### kv (incompressible UUID 약 20k context, 150 turns, context full = r=1.0)
random UUID 컨텍스트는 whole-position drop이 부적합 → context는 보존(`--comp_ratio 1.0 --light_drop_ratio 0`)하고 **turn 누적만** 압축 → multi-turn memory 효과 단독 검증.
```bash
conda run -n jk python run_scbench.py -m qwen3-4b -d scbench_kv_short \
    --method ours --comp_ratio 1.0 --light_drop_ratio 0 --turn_ratio 0.4 \
    --turns 150 --consec_zero_stop 5
conda run -n jk python run_scbench.py -m qwen3-4b -d scbench_kv_short \
    --method kvzip --ratio 1.0 --turns 150 --consec_zero_stop 5
```

각 실행의 raw record는 `results/scbench/<...>.json` (턴별 trace 포함), flat 요약은 `results/scbench_runs.jsonl`.

---

## 6. 결과 집계 (paper 표 자동 생성)

```bash
conda run -n jk python aggregate.py
```

출력:
- `results/summary_niah_grid.md` — model × ctx_len 별 **prefill peak (GB) 표 + 정확도 표 + per-position 히트맵** (paper Table 1/2/3).
- `results/summary_scbench.md` — method × task 별 **prefill / turn peak / cache / 정확도 / per-turn growth** (paper Table 4/5/6).
- `results/summary_niah_grid.csv`, `results/summary_scbench.csv` — long-format (플롯/분석용).

paper.tex의 모든 표 숫자는 이 두 markdown 파일에서 직접 발췌.

---

## 7. 주요 CLI 인자 (`args.py` + `run_scbench.py`)

| 인자 | 의미 | 기본 |
|---|---|---|
| `--method` | full / kvzip / streamllm / ours | ours |
| `--level` | per_token (dense, multi-turn 필수) / per_head (KVzip varlen finalize) | per_token |
| `--comp_ratio` | 두 메서드에 동일 적용되는 retained 비율 $r$. kvzip prune ratio = $r$; ours는 $B_f = \text{round}(rL)$로 자동 derive. `<0`이면 off (옛 명시 budget 사용) | -1 |
| `--intermediate_budget` (B_i) | heavy-prune 후 segment당 유지 토큰 수 (comp_ratio가 자동 derive) | 2048 |
| `--working_max` | heavy-prune 트리거 (0이면 `2*B_i`) | 0 |
| `--final_budget` (B_f) | 누적 committed 캡 | 8192 |
| `--recent_window` | prune 시 절대 안 버리는 최신 토큰 | 256 |
| `--chunk` | prefill + scoring chunk 크기 (NIAH grid 기본은 512) | 2048 |
| `--token_agg` | head/layer score 집계 (mean / max) | mean |
| `--light_drop_ratio` | chunk 별 light drop 컷오프 (`score < ratio*chunk_max`) | 0.05 |
| `--no_rescore_working` | heavy-prune 전 working 재스코어 비활성 | off |
| `--head_ratio` | per_head finalize 유지 비율 | 0.5 |
| `--use_predict_prompt` / `--predict_prompt_version {1,2}` | 중간 prune을 predict-self 프롬프트로 (최종 compaction은 항상 repeat) | off / 1 |
| `--predict_target {self,next_chunk}` | predict 신호 종류 (next_chunk는 sink/recent 보호) | self |
| `--combine_repeat` / `--combine_mode {max,wsum}` / `--combine_alpha` | 중간 점수 = combine(repeat, predict) | off / max / 0.5 |
| `--sink_tokens` | predict-next boundary 보호용 leading sink 토큰 수 | 4 |
| `--ratio` | kvzip baseline의 retained ratio (`--comp_ratio` 미지정 시 사용) | 0.3 |
| `--turn_ratio` *(scbench)* | DYNAMIC per-turn 압축 비율. `<0`이면 legacy fixed-cap | 0.4 |
| `--turn_base_budget` *(scbench)* | turn 압축 base budget. `<0`이면 context $B_f$ 사용 | -1 |
| `--turn_cap_mult` *(scbench legacy)* | fixed-cap 모드의 turn region 캡 배수 | 2 |
| `--turn_rescore_every` *(scbench legacy)* | full repeat 재스코어 주기 (cap reprunes) | 8 |
| `--consec_zero_stop` *(scbench)* | acc=0 연속 N turn이면 정지 | 5 |
| `--max_new` | answer 생성 max new tokens | 64 |
| `--device` / `--tag` | GPU id / 실행 태그 | 0 / None |

`--comp_ratio` + `--turn_ratio`가 본 실험에서 권장하는 **유일한 두 손잡이**다. 나머지는 ablation용.

---

## 8. Peak 측정 방식

`ric/mem.py`의 `peak_gb` / `reset_peak` 가 모든 실험에서 호출된다.

- Peak = `torch.cuda.max_memory_allocated()` (KV 저장 + transient attention activation 모두 포함)
- 매 셀 시작에서 `reset_peak()` (= `torch.cuda.reset_peak_memory_stats()` + `gc.collect()` + `torch.cuda.empty_cache()`)
- 전 forward는 bf16, attention backend는 flash-attn 2.8.3
- NIAH context는 (ctx_len, depth)당 한 번 생성된 token ids를 모든 (method, ratio) 셀에 동일 입력 → 메서드 간 차이는 저장 KV만의 함수

paper §4.1의 "Memory protocol" 단락과 동기.

---

## 9. Citing

본 레포의 baseline 및 기반:
- **KVzip** (Kim et al. 2026) — reconstruction-based importance + per-head varlen prune. `kim2026kvzip`.
- 그 외 KV cache compression / loading 라인업은 `paper.tex`의 references 참고 (StreamLLM, SnapKV, PyramidKV, H2O, SCOPE, EpiCache, Quest, Squeezed Attention 등).
