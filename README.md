# Reducing-Initial-Cache (RIC)

**스트리밍 KV-cache 압축으로 prefill의 *peak* 메모리를 줄이는 방법.**

KVzip / SnapKV / PyramidKV 등 기존 prefill 압축 기법은 **긴 context의 KV를 전부 만든 뒤(materialize) 압축**한다. 따라서 압축을 해도 *peak* 메모리는 full KV 그대로이고, 같은 GPU에서 감당 가능한 초기 context 길이의 상한은 baseline과 동일하다(오히려 KVzip은 scoring forward 때문에 peak가 더 높다). 이 프로젝트(논문 `paper.tex`의 방향 1)는 **prefill 도중 압축을 interleave**해서 물리적으로 저장되는 KV를 항상 budget 이하로 묶어 peak를 낮춘다. 결과적으로 12GB GPU에서 baseline보다 더 긴 초기 context를 OOM 없이 삼킬 수 있다.

> 상태: **구현 완료, 아직 실행 안 함.** 모델 가중치(llama3.2 등)는 로컬에 없으며 게이트될 수 있다. 아래 명령으로 직접 실행해야 한다. 코드 정적 syntax 체크는 통과(`python -m py_compile`).

---

## 1. 핵심 아이디어 (Method B)

```
for chunk in context_chunks:
    prefill(chunk)                      # dense append (KV가 budget까지 자람)
    score(chunk)                        # KVzip reconstruction scoring (chunk-local)
    if physical_ctx > budget_max:       # budget 초과시에만
        (optional) re-score cached ctx  #  --rescore
        compact -> budget_target        #  중요한 토큰만 물리적으로 남김
```

- **왜 가능한가:** KVzip의 importance scoring은 *chunk-local*이다(각 chunk를 자기 자신의 "repeat" reconstruction으로 점수화). 따라서 전체 context를 만들지 않아도 갓 prefill한 chunk를 즉시 점수화할 수 있다.
- **왜 peak가 줄어드는가:** 물리적으로 저장되는 context KV가 항상 `budget_max + chunk` 이하로 유지된다. baseline은 full KV를 만들어야 하므로 같은 길이에서 OOM.
- **사용자의 직관 그대로:** 초기에는 거의 안 버리다가(낮은 threshold) budget을 넘기면 reconstruction을 다시 돌려 더 세게 추려낸다(`--rescore` + `budget_target`).

### 두 가지 압축 단위 (`--level`)
- **`per_token` (기본, dense):** head 전체에 공유되는 토큰 position 단위로 evict. cache가 dense로 유지돼 재-scoring이 쉽고 실제 메모리가 해제된다. prefill 내내 normal attention 사용.
- **`per_head` (KVzip 방식):** streaming prefill은 per_token과 동일(peak를 budget으로 bound)하게 돌고, **마지막에** KVzip의 per-head non-uniform eviction(`prune`)을 dense cache에 적용해 varlen으로 만든다 → answer 단계의 메모리가 더 낮아지고 head별 선택이 적용된다. KVzip의 검증된 코드(`prune`/`prepare_init`)를 그대로 재사용.

### 불변식 (중요)
`_seen_tokens`는 항상 **logical** 길이(sink + 지금까지 처리한 모든 context 토큰)를 추적하고, 물리 cache 길이와 분리된다. 살아남은 key는 원래 RoPE 위치를 유지하고(중간에 구멍이 생김), 이후 chunk/query는 logical 위치로 RoPE를 받는다. 이는 **KVzip이 post-prune inference에서 이미 의존하는 근사**와 동일하다. 토큰을 재배열하지 않고 *drop*만 하므로 물리 배열 순서 = logical 순서 → flash causal attention이 그대로 성립한다.

---

## 2. 코드 구조

```
Reducing-Initial-Cache/
├── ric/
│   ├── __init__.py        # KVzip을 sys.path에 주입(설치 없이 재사용)
│   ├── mem.py             # peak 메모리 측정 + OOM-safe 러너
│   ├── stream_cache.py    # StreamingCache(EvictCache): 압축, logical/physical 분리, finalize
│   ├── stream_prefill.py  # streaming prefill 루프 + scoring + multi-turn
│   └── baselines.py       # full / kvzip / streamllm / ours 디스패치
├── args.py                # 공용 CLI
├── run_niah.py            # 실험 1: NIAH 길이 사다리 OOM
├── run_scbench.py         # 실험 2: SCBench short-ctx multi-turn OOM
└── README.md
```

KVzip 코드는 `../KVzip`에서 import한다(설치 X — KVzip의 pyproject가 torch 2.3/numpy 1.26으로 다운그레이드시키므로 일부러 설치하지 않음). 다른 위치면 `KVZIP_DIR` 환경변수로 지정.

---

## 3. 환경

**이 레포는 KVzip을 런타임 의존성으로 import한다.** KVzip을 형제 디렉토리로 clone하거나 `KVZIP_DIR` 환경변수로 지정해야 한다.

- **새 서버에서 처음 세팅**: → **[`SETUP.md`](SETUP.md)** 를 따른다(단계별 매뉴얼 + 함정 정리). 핵심: python 3.12 + `requirements.txt`, 서버 CUDA에 맞는 torch, 서버 GPU arch로 flash-attn 빌드, `KVzip/csrc/build.py`의 arch 수정 후 커널 빌드, **transformers는 반드시 4.51.3**, KVzip은 `pip install -e .` 하지 말 것.
- **소스 서버(`jk`) 기준 동작 버전**: torch 2.8.0 / transformers 4.51.3 / flash-attn 2.8.3 / RTX 3080 Ti(sm_86) / CUDA 12.9. 정확한 freeze는 `env/jk-conda-full.yml`, `env/jk-pip-freeze.txt`.

항상 `conda run -n jk ...` 또는 `conda activate jk` 후 실행.

---

## 4. 실행법

### 실험 1 — NIAH 길이 사다리 (peak 메모리 / OOM 한계)
```bash
conda run -n jk python run_niah.py \
    -m llama3.2-3b --method ours --level per_token \
    --budget_max 8192 --budget_target 4096 --chunk 2048 \
    --ctx_lens 4000,8000,16000,32000,48000,64000 --depth 50
```
각 길이마다 NIAH context를 만들어 prefill하며 **peak GPU 메모리**를 측정하고, OOM이 안 났으면 needle 질문에 답해 정확도를 본다. prefill OOM이 나면 그 사다리는 중단(더 길면 당연히 OOM). 결과 `results_niah/`에 저장.

baseline과 비교(같은 인자로 `--method`만 바꿈):
```bash
for M in full kvzip streamllm ours; do
  conda run -n jk python run_niah.py -m llama3.2-3b --method $M --level per_token \
     --budget_max 8192 --budget_target 4096 --ctx_lens 4000,8000,16000,32000,48000,64000 --tag cmp
done
```
per_head도 동일하게 `--method ours --level per_head --head_ratio 0.5`.

### 실험 2 — SCBench short-context multi-turn
```bash
conda run -n jk python run_scbench.py \
    -m llama3.2-3b -d scbench_kv_tiny --method ours \
    --budget_max 8192 --budget_target 4096 --turns 200 --turn_budget 2048
```
짧은 공유 context를 한 번 prefill한 뒤, 질문을 cycling하며 multi-turn 대화를 누적(`update_cache=True`)한다. ours는 누적분(turn KV)도 bound → 오래 버팀. `full`/`kvzip`은 bound 없이 누적 → 먼저 OOM. 살아남은 turn 수 / 턴별 메모리 / 정확도를 기록.

### 주요 인자
| 인자 | 의미 | 기본 |
|---|---|---|
| `--method` | full / kvzip / streamllm / ours | ours |
| `--level` | per_token / per_head | per_token |
| `--budget_max` | 압축 trigger(물리 context 토큰 수) | 8192 |
| `--budget_target` | 압축 후 남길 토큰 수 | 4096 |
| `--recent_window` | prefill 중 절대 안 버리는 최신 토큰 | 256 |
| `--chunk` | prefill+scoring chunk 크기 | 2048 |
| `--token_agg` | head/layer score 집계(mean/max) | mean |
| `--rescore` | compaction 때 cached context 재-scoring | off |
| `--head_ratio` | per_head finalize 남길 비율 | 0.5 |
| `--ratio` | kvzip baseline prune 비율 | 0.3 |

### 결과 기록 & 집계 (paper.tex 작성용)
모든 실행은 **자동으로 정확·상세하게 기록**된다(`ric/record.py`):
- `results/niah/<model>_<method>_<level>_<tag>_<timestamp>.json` — 한 실행의 전체 detail(환경/GPU/모델 config/KV bytes-per-token/인자/cfg/모든 ctx_len row/생성된 답변/compaction 수/peak·cache 메모리/소요시간). **timestamp가 붙어 덮어쓰지 않음** → 여러 번 돌려도 다 남음.
- `results/scbench/<...>.json` — 동일하게 prefill 통계 + 턴별 trace(메모리/정확도/답변).
- `results/niah_runs.jsonl`, `results/scbench_runs.jsonl` — 실행마다 한 줄씩 누적되는 flat 요약(집계용).

여러 실행을 돌린 뒤 **집계 → paper-ready 표** 생성:
```bash
conda run -n jk python aggregate.py
```
출력:
- `results/summary_niah.md` — method×ctx_len **prefill peak 메모리 표**, 정확도 표, "OOM 전 최대 context" headline 표.
- `results/summary_scbench.md` — method별 생존 turn 수 / 평균 정확도 / 메모리 표.
- `results/summary_niah.csv`, `results/summary_scbench.csv` — long-format(플로팅/분석용).

→ 실험을 돌리고 `aggregate.py`만 실행하면 `summary_*.md`/`.csv`가 갱신되고, 그 수치로 paper.tex를 작성하면 된다.

---

## 5. 실험 설계 / 공정성

- **Peak 측정:** `torch.cuda.max_memory_allocated()` (transient attention + 저장 KV 모두 포함). 매 실행 전 `reset_peak`.
- **공정한 baseline:** `full`/`kvzip`도 우리와 같은 `--chunk`를 prefill_chunk_size로 사용 → transient는 동일, **저장 KV만 차이**나게 함. `kvzip`은 full prefill + scoring + per-head prune이라 peak ≥ full(thesis가 겨냥하는 지점).
- **모델:** 12GB면 `llama3.2-3b`(가중치 ~6.4GB) 우선. OOM 한계가 너무 빨리 오면 `llama3.2-1b` / `qwen3-1.7b`로 헤드룸 확보.

기대 결과: full/kvzip은 중간 길이(3B면 대략 30~40k)에서 prefill OOM, ours는 budget 덕에 훨씬 길게 진행. SCBench는 누적 turn에서 ours만 평탄한 메모리로 오래 생존.

---

## 6. 알려진 한계 / 아직 검증 안 된 부분

코드는 **정적 syntax만 통과**했고 실제 실행은 안 했다. 런타임에서 점검/디버깅이 필요한 지점:

1. **mid-prefill eviction 후의 dense 재-prefill** — `_seen_tokens`(logical)와 물리 cache 길이가 어긋난 상태에서 다음 chunk를 prefill하는 경로는 KVzip이 직접 시험하지 않은 사용법이다(KVzip은 prefill 다 끝낸 뒤에만 prune). 위 불변식상 맞게 설계했으나 첫 실행 시 RoPE/positional 동작을 확인할 것.
2. **gappy re-scoring(`--rescore`)** — 압축으로 구멍 난 cache를 다시 점수화할 때 살아남은 key는 원래 RoPE를 유지. 근사이며(품질은 NIAH/SCBench가 드러냄) 기본 off.
3. **per_head + multi-turn** — varlen cache는 dense turn-eviction과 호환되지 않아 SCBench는 per_token으로 강제(코드가 경고 후 fallback). per_head multi-turn(varlen-aware turn 관리)은 향후 과제.
4. **compaction 순간 transient spike** — `torch.cat`으로 새 텐서를 만들므로 압축 순간 일시적으로 old+new가 공존(≈2×budget). bound 안이지만 `index_select` in-place로 최적화 여지.
5. **per_head의 메모리 이점** — 현재 per_head는 prefill peak를 per_token과 *같은* budget으로 bound하고, finalize에서만 추가로 줄인다. prefill 도중부터 head별로 다른 길이를 저장하는 "진짜 varlen 스트리밍"(더 낮은 peak)은 fresh-chunk를 varlen에서 dense로 추출해 점수화하는 기법이 필요하며 향후 과제. (메모리 노트 `ric-implementation-plan` 참고)

---

## 7. 다음 단계
1. tiny 설정으로 smoke test: `-m llama3.2-1b --ctx_lens 2000,4000 --budget_max 2048 --budget_target 1024` 가 도는지 + needle 정확도 확인.
2. per_token NIAH 사다리 full-sweep → baseline 대비 OOM 한계 그래프.
3. per_head NIAH 비교(정확도/메모리).
4. SCBench multi-turn 생존 turn 수 비교.
5. 결과를 `paper.tex` Problem/Approach/Experiment에 반영(peak prefill 메모리 프레이밍).
