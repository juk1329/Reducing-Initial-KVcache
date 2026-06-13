# RIC — Results Notes (staging for paper.tex)

Staging doc consolidating experiment results + method evolution. **Not** the paper; paper.tex
is written once, after SCBench. Numbers sourced from `results/summary_niah_grid.md` +
`results/niah_grid/*full_sink*.json` (NIAH) and (later) `results/summary_scbench.md`.

Models: `qwen3-4b` (clean retriever — accuracy story), `llama3.2-3b` (secondary; noisy NIAH
acc ≥8k but confirms the memory story). GPU: single 24 GB (L4, 22.5 GB usable).

---

## Prompt study — peak 감소용 중간(intermediate) 스코어링 prompt 비교

peak를 줄이는 압축에서 "어떤 prompt로 KV 중요도를 매겨 무엇을 남길지"를 비교. **핵심: peak/메모리는 예산(ratio)이 결정하고 prompt가 아님 — prompt는 같은 예산에서 어떤 토큰을 남기는지(=정확도)를 결정.** 변형: `repeat`(재구성), `predict-self`(프롬프트 "Predict the entire/upcoming context:" v1/v2), `predict-next`(다음 chunk 타깃; NIAH 전용, 멀티턴 turn엔 '다음 turn' 없어 부적용), `combine`(repeat+predict). 최종 압축은 항상 repeat.

**NIAH (qwen3-4b, ctx=8000) — 변형 6종 모두 보유** (`results/summary_niah_grid.md`):
- r≥0.4: repeat = combine = predict-self(v1·v2) = 1.0 (대부분). predict-next는 중앙 needle 붕괴(0.45~0.55).
- r=0.2(공격적): repeat 0.73 > predict-self v1 0.59 / v2 0.64 > predict-next 0.45; **combine 1.0**(repeat 신호로 회복). 
- 결론(NIAH): predict 단독은 repeat보다 나쁨(특히 predict-next는 needle 비예측성으로 붕괴), combine이 repeat 수준 회복. **repeat가 강한 기준선.**

**SCBench (qwen3-4b) — 변형 5종 보유** (predict-next 제외; `results/summary_scbench.md`):

kv_short(@1.0, turn_ratio 0.4, 150턴) — acc / turn-peak / cache증가율(GB/턴):
| kvzip | repeat | predict_v1 | predict_v2 | combine |
|---|---|---|---|---|
| 1.000 / 14.66 / .0195 | **1.000 / 13.10 / .0081** | 0.907 / 12.67 / .0054 | 0.907 / 12.67 / .0054 | **1.000 / 13.10 / .0081** |

summary_short(@0.4, turn_ratio 0.4, 100턴):
| kvzip | repeat | predict_v1 | predict_v2 | combine |
|---|---|---|---|---|
| 0.487 / 9.86 / .0121 | **0.530 / 9.14 / .0040** | 0.435 / 9.07 / .0051 | 0.440 / 9.05 / .0040 | 0.448 / 9.01 / .0040 |

**종합 결론 (prompt study):**
1. **메모리/peak는 prompt가 아니라 예산이 결정** — 모든 ours 변형이 비슷한 peak·증가율로 kvzip을 하회. predict 계열이 cache가 약간 더 작은 건 정확도를 희생해 더 적게 남기기 때문(트레이드오프).
2. **정확도는 repeat가 최선** — kv: repeat·combine 1.0 vs predict 0.907; summary: repeat 0.530 vs predict/combine 0.44~0.45 (kvzip 0.487). NIAH·SCBench 일관.
3. **predict 단독은 retrieval/summary에 불리**(예측 신호 ≠ 보존해야 할 정보). **combine은 NIAH 공격적 영역에서 repeat 수준 회복**하나 그 외엔 repeat와 비슷하거나 약간 낮음.
4. 따라서 "peak를 줄이며 성능 방어"의 prompt 선택은 **repeat가 기본 권장**; predict/combine은 ablation으로 보고(특히 NIAH r=0.2에서 combine의 회복 효과가 novelty 포인트).

---

## Experiment 1 — NIAH (initial-context prefill peak)

**Setup.** Needle-in-a-Haystack. Fixed contexts **500 / 2000 / 8000** tokens; **11 needle
positions** (depth 0,10,…,100%); compression ratio **r ∈ {0,0.2,0.4,0.6,0.8,1.0}** applied
identically to both methods. **All per-head** (varlen) finalize, matching KVzip → fair. The
context for each (ctx_len, needle position) is generated **once** and the identical token ids
fed to every method/ratio (fixes a lossy re-encode mismatch). Metric: needle accuracy (mean
over 11 positions) + prefill peak memory.

### Headline: peak memory
- **KVzip prefill peak is ratio-independent** — it materializes the full-context KV before
  pruning, so peak is identical for every r: **10.68 GB (qwen) / 8.46 GB (llama) at 8000 tokens**.
  Compression reduces only the *stored* cache, not the initial peak.
- **Ours bounds the stored context during prefill**, so peak scales with r and is **always below
  KVzip**: ours/kvzip = **0.79→0.88** (qwen) and **0.79→0.89** (llama) across r at 8000.
- **The gap grows with context length** (ours/kvzip at r=0.4): 500 → 1.00 (tie; single-chunk
  scoring-transient floor), 2000 → 0.87, 8000 → 0.86. Extends to OOM-reach at longer contexts.

Peak (GB) @ ctx=8000, qwen3-4b:
| r | KVzip | ours | ours/KVzip |
|---|---|---|---|
| 0.2 | 10.68 | 8.78 | 0.82 |
| 0.4 | 10.68 | 9.20 | 0.86 |
| 0.6 | 10.68 | 9.07 | 0.85 |
| 1.0 | 10.68 | 9.42 | 0.88 |

### Accuracy (qwen3-4b — clean)
- **r ≥ 0.4: ours = kvzip = 1.0** at every context length and needle position. So ours matches
  KVzip retrieval at ~14% lower peak (r=0.4, 8000).
- **r = 0.2** (most aggressive) is the only regime with a gap: at 8000, ours 0.73 vs kvzip 1.0
  (streaming occasionally evicts a mid-context needle; recency-protected ends stay perfect).
- r = 0.0 → 0 (nothing retained).
- **llama**: peak story identical; accuracy noisy/non-monotone ≥8000 (kvzip itself 0.91/0.27/1.0
  at r=0.2/0.4/0.6) → 3B retriever unreliable, so qwen carries the accuracy story.

---

## Method evolution (the "ours" variants, NIAH per-head grid)

All ours variants bound prefill peak the same way (peak ≈ ours, < kvzip). They differ only in
the **intermediate scoring signal** used for the peak-reducing prunes; the **final** compaction
always uses the **repeat** (reconstruction) prompt, like KVzip.

1. **ours (repeat-only).** Repeat prompt for every prune. Baseline ours. r≥0.4 → acc 1.0.
2. **ours_predict_v1 / v2 (predict-self prompt).** Intermediate prunes scored by a *predict*
   prompt on the chunk itself ("Predict the entire context:" / "Predict the upcoming context:");
   final stays repeat. Result: **no gain, slightly worse** than repeat (qwen 8000/r0.2: 0.68 vs
   0.73), with a bias toward dropping the earliest (q=0.0) needle. v1≈v2.
3. **ours_predict_next (predict target = real next chunk) + boundary protection.** Score each
   chunk by how much the *real next chunk* attends to it (true causal-prediction signal); keep
   first chunk = sink, last chunk = recent. **Result: middle needle collapses** (q=0.5 acc 0 even
   at r=0.4) — the needle isn't predictive of the haystack continuation, so predict-next gives it
   low importance. Genuine negative: predictive importance ≠ retrieval importance.
4. **ours_combine (predict-next + repeat, the fix).** Same predict-next setup, but intermediate
   score = **max(repeat-self, predict-next)** per (layer,head,position); final stays repeat. The
   repeat term restores the needle. **Recovers retrieval to ours's level** while keeping the
   peak advantage and the predictive signal. 14 grid cells fixed vs predict_next.
5. **Small-sink tuning (current default).** Boundary protection capped to a small StreamingLLM
   sink (`sink_tokens=4`) + `recent_window` (was whole first/last chunk, which ate the budget at
   aggressive r). **Clear win, no regression:** e.g. ours_combine acc 0.45→0.86 (qwen 500/r0.2),
   0.45→1.0 (llama 2000/r0.2); peak unchanged.

### Final method (ours_combine, tuned)
- **Peak**: == ours, 11–21% below kvzip @8000 (kvzip ratio-independent). Preserved.
- **qwen3-4b acc**: **r≥0.4 → ours_combine = ours = kvzip = 1.0 everywhere.** r=0.2 only: 0.64–0.86
  (≈ ours at 8000; both < kvzip 1.0 = streaming's intrinsic limit at the extreme ratio).
- **llama3.2-3b acc**: r≥0.2 → 1.0 everywhere, matching/beating both baselines in the noisy long
  cells (8000/r0.4: combine 1.0 ≫ ours 0.45 > kvzip 0.27).
- Defaults: `sink_tokens=4, combine_mode=max, recent_window=64, chunk=512, head_ratio_target=0.5`.
- Takeaway for the paper: ours_combine fully matches KVzip/ours retrieval at the practical
  compression regime while keeping ours's bounded prefill peak, and carries the novelty
  (predictive + reconstructive importance, bounded peak, sink stability).

Data: `results/summary_niah_grid.md`, `results/niah_grid/*full_sink*.json`,
`results/summary_niah_grid.csv`.

---

## Experiment 2 — SCBench (멀티턴) — 동적 per-turn 압축(turn_ratio)

**세팅.** 초기 long context를 한 번 prefill한 뒤 멀티턴(질문 순환). ours는 각 (query+answer) 턴을
chunk처럼 처리: 등록 → score(ours=repeat, ours_combine=combine(repeat,predict-self)) → **그 턴을
`round(turn_ratio·Lₜ)`개로 동적 압축**(턴 길이 비례) 후 commit. 즉 초기 context를 ratio r로 압축한
철학을 턴에도 적용해 **세션 전체(context+turns)를 동일 비율로 균일 압축**. 초기 context는 보호된 prefix.
kvzip = per-head context prune + 턴은 무압축 누적(baseline). qwen3-4b, per_token(턴 append 위해 dense).
지표: substring/token-overlap(kv는 공식 `label in pred`; summary는 token-overlap, 정식 ROUGE 아님).
중단: completed / oom / acc_collapse(5연속 0). 데이터: `results/summary_scbench.md`,
`results/scbench/*expA/expB*.json`.

### Exp B — `scbench_kv_short` (kv@comp_ratio 1.0, turn_ratio 0.4, 150턴) — 멀티턴 메모리 방어 핵심
context = 무작위 UUID 키-값(per_token 압축 불가 → context는 full 유지=acc 1.0; **턴 누적**만 분리 검증).

| method | acc | prefill peak | turn-phase peak | cache(턴0→끝) | 증가율(GB/턴) |
|---|---|---|---|---|---|
| kvzip | 1.000 | 13.11 | 14.66 | 3.60→6.50 | **0.0195 (무압축 누적)** |
| ours | 1.000 | 13.17 | **13.10** | 3.60→**4.80** | **0.0081 (≈0.42×)** |
| ours_combine | 1.000 | 13.17 | **13.10** | 3.60→**4.80** | **0.0081** |

→ ours는 턴을 0.4 비율로 압축 → **cache 증가율이 kvzip의 0.42배**(0.0081 vs 0.0195), 150턴 후 cache
4.80 vs 6.50GB, turn-peak 13.10 < 14.66, **정확도 1.0 동일**. context가 full이라 옛 턴 drop이 정확도에
무해(답은 context에서 나옴). (context 무압축이라 prefill peak는 ours≈kvzip — 예상대로 이점 없음;
이점은 턴 누적 억제.)

### Exp A — `scbench_summary_short` (comp_ratio 0.4, turn_ratio 0.4, 100턴) — 압축+정확도 방어
압축 가능한 NL context(per_token 작동). context를 0.4로 압축 + 턴도 0.4로 압축.

| method | acc | prefill peak | turn-phase peak | cache(턴0→끝) | 증가율(GB/턴) |
|---|---|---|---|---|---|
| kvzip | 0.487 | 10.91 | 9.86 | 0.60→1.80 | 0.0121 |
| ours | **0.530** | **10.46** | **9.14** | 0.60→**1.00** | **0.0040 (≈0.33×)** |
| ours_combine | 0.448 | 10.67 | **9.01** | 0.50→**0.90** | **0.0040** |

→ **ours/ours_combine가 prefill peak·turn-phase peak·저장 cache·증가율 모두 kvzip보다 낮으면서 정확도
방어**(ours 0.530 ≈ kvzip 0.487 ≈ full 0.53; ours_combine 0.448로 소폭 낮음). NIAH와 동일하게 "초기
context의 peak를 줄이며 성능 방어"가 멀티턴에서도 성립.

### 핵심 발견 (동적 turn_ratio가 왜 더 나은가)
1. **고정 cap의 문제(이전 검증):** cap=2·B_f가 "작게 압축한 summary context"보다 커서 멀티턴 cache가
   압축분을 도로 키우고, cap 영역 전체를 주기적 repeat 재스코어하는 transient 때문에 **summary
   turn-phase peak가 kvzip보다 높았음(11.27>9.86)**. (성장 cap m·B_f는 누적을 그대로 따라가 bounding도 안 됨.)
2. **동적 per-turn 압축이 해결:** 각 턴을 그 길이의 turn_ratio로 즉시 압축 → (a) **turn마다 작은
   연산만**(큰 cap 재스코어 없음) → transient 제거 → **turn-peak가 kvzip 아래로**(summary 9.14<9.86,
   kv 13.10<14.66). (b) 세션 전체가 ratio r 균일 압축 → cache 증가율 = r × kvzip(상수배 절감), OOM
   1/r배 지연. (c) 정확도 방어.
3. **per_token은 무작위-KV context 압축 불가**(통째 drop으로 키-값 손실) → kv는 context를 full(comp 1.0)
   로 두고 턴만 압축. per_head-context+dense-turns 하이브리드가 근본 해법(향후 과제).
4. trade-off: 동적은 bounded plateau가 아니라 r×기울기의 선형 증가(고정 cap=4.20 plateau보다 kv에선 약간
   큰 4.80) — 대신 transient가 작아 turn-peak↓ 이고 "균일 r 압축" 철학이 일관됨.
