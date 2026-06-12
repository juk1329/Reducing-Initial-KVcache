#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Experiment 1 - NIAH length ladder: "how long an initial context can each method
# ingest before OOM?"
#
# For a sweep of context lengths we generate a Needle-in-a-Haystack example,
# prefill it with the chosen method while measuring PEAK GPU memory, and (if it
# did not OOM) answer the needle question to check the context was preserved.
#
# Expected story on a 12GB GPU: `full` and `kvzip` OOM at a moderate length
# (they materialize the full-context KV); `ours` (per_token / per_head) keeps
# going to much longer contexts because peak is bounded by the budget.
#
# This script DOES NOT auto-run; invoke it explicitly, e.g.:
#   conda run -n jk python run_niah.py -m llama3.2-3b --method ours --level per_token \
#       --budget_max 8192 --budget_target 4096 --ctx_lens 4000,8000,16000,32000,48000
# ------------------------------------------------------------------------------
import argparse
import gc
import json
import os

import torch

import time

from args import add_common_args, cfg_from_args
import ric
from ric import KVZIP_DIR
from ric.baselines import build_prefill_fn
from ric.mem import run_oom_safe, total_gb, peak_gb, reset_peak
from ric.record import capture_env, save_run
from ric.stream_prefill import answer, derive_ric_budgets
from dataclasses import replace

from model import ModelKVzip
from data.needle import NeedleHaystackData

NEEDLE_GOLD_KEYWORDS = ["sandwich", "dolores park"]  # NIAH answer: "Eat a sandwich and sit in Dolores Park..."


def make_niah(model, ctx_len, depth):
    hay = os.path.join(KVZIP_DIR, "data", "needle", "PaulGrahamEssays")
    nh = NeedleHaystackData(
        model.tokenizer, haystack_dir=hay,
        context_lengths=[ctx_len], final_context_length_buffer=0,
    )
    data = nh.generate_context_qa(ctx_len, depth)
    return data["context"], data["question"][0], data["answers"][0]


def score_answer(ans: str) -> float:
    a = ans.lower()
    return sum(k in a for k in NEEDLE_GOLD_KEYWORDS) / len(NEEDLE_GOLD_KEYWORDS)


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--ctx_lens", default="4000,8000,16000,24000,32000,48000,64000",
                   help="comma-separated context lengths to sweep")
    p.add_argument("--depth", type=int, default=50, help="needle depth percent")
    a = p.parse_args()
    cfg = cfg_from_args(a)
    ctx_lens = [int(x) for x in a.ctx_lens.split(",")]

    # compression-ratio sweep mode: a single retained fraction r applied to both methods.
    use_cr = a.comp_ratio is not None and a.comp_ratio >= 0.0
    eff_kvzip_ratio = a.comp_ratio if use_cr else a.ratio
    print(f"GPU total ~{total_gb(a.device):.1f}GB | model={a.model} method={a.method} "
          f"level={a.level} "
          + (f"comp_ratio={a.comp_ratio} (final_budget=round(r*L), B_i/working_max derived) "
             if use_cr else
             f"B_i={a.intermediate_budget} working_max={cfg.working_max} B_f={cfg.final_budget} ")
          + f"chunk={a.chunk}")
    model = ModelKVzip(a.model, kv_type="evict")
    env = capture_env(model, a.device)
    print(f"env: {env.get('gpu_name')} {env.get('gpu_total_gb')}GB | "
          f"KV {env.get('kv_gb_per_1k_tokens')} GB/1k tok | "
          f"torch {env.get('torch')} tf {env.get('transformers')} fa {env.get('flash_attn')}")
    # For ours under a comp_ratio sweep the budgets depend on ctx_len, so the prefill fn is
    # rebuilt per length (below). Otherwise build it once here.
    ours_cr = use_cr and a.method in ("ours", "ric")
    prefill_fn = None if ours_cr else build_prefill_fn(a.method, model, cfg, ratio=eff_kvzip_ratio)

    rows = []
    for L in ctx_lens:
        print("=" * 80, f"\n[ctx_len={L}] building NIAH ...", flush=True)
        ctx, q, gold = make_niah(model, L, a.depth)
        ctx_ids = model.encode(ctx)
        true_len = ctx_ids.shape[1]

        # effective budgets for this length (derived from comp_ratio for ours)
        if ours_cr:
            fb, bi, wm = derive_ric_budgets(a.comp_ratio, true_len, a.chunk, a.recent_window)
            cfg_L = replace(cfg, final_budget=fb, intermediate_budget=bi, working_max=wm)
            prefill_fn = build_prefill_fn("ours", model, cfg_L, ratio=eff_kvzip_ratio)
            eff_ib, eff_wm, eff_fb = cfg_L.intermediate_budget, cfg_L.working_max, cfg_L.final_budget
        else:
            eff_ib, eff_wm, eff_fb = cfg.intermediate_budget, cfg.working_max, cfg.final_budget

        t0 = time.time()
        kv, info = run_oom_safe(lambda: prefill_fn(ctx_ids), device=a.device)
        row = {
            "ctx_len": L, "true_len": int(true_len),
            "method": a.method, "level": a.level,
            "comp_ratio": (a.comp_ratio if use_cr else None),
            "kvzip_ratio": (eff_kvzip_ratio if a.method == "kvzip" else None),
            "intermediate_budget": eff_ib, "working_max": eff_wm,
            "final_budget": eff_fb,
            "prefill_status": info["status"],
            "prefill_peak_gb": info["peak_gb"],
            "prefill_alloc_gb": info.get("alloc_gb"),
            "prefill_sec": round(time.time() - t0, 2),
            "answer_peak_gb": None, "cache_gb": None,
            "num_compactions": None, "num_segments": None,
            "peak_phys_ctx": None, "logical_ctx_len": None,
            "acc": None, "answer": None, "question": q, "gold": gold,
            "error": info.get("error"),
        }

        if info["status"] == "ok":
            # capture compaction stats (StreamingCache only; baselines lack them)
            row["num_compactions"] = getattr(kv, "num_compactions", None)
            row["num_segments"] = getattr(kv, "num_segments", None)
            row["peak_phys_ctx"] = getattr(kv, "peak_phys_ctx", None)
            row["logical_ctx_len"] = getattr(kv, "logical_ctx_len", None)
            row["cache_gb"] = float(kv._mem())
            # answer phase (separate peak; the headline OOM wall is prefill)
            reset_peak(a.device)
            try:
                ans = answer(model, kv, q, max_new=a.max_new, update_cache=True)
                row["acc"] = score_answer(ans)
                row["answer"] = ans[:300]
                row["answer_peak_gb"] = peak_gb(a.device)
                print(f"  -> OK prefill_peak={info['peak_gb']:.2f}GB cache={row['cache_gb']:.2f}GB "
                      f"compactions={row['num_compactions']} acc={row['acc']:.2f} | {ans[:80]!r}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                row["answer"] = f"ANSWER_ERROR: {str(e)[:200]}"
                print(f"  -> prefill OK but answer failed: {e}", flush=True)
        else:
            print(f"  -> {info['status'].upper()} at peak~{info['peak_gb']}", flush=True)

        rows.append(row)
        del kv
        gc.collect()
        torch.cuda.empty_cache()
        # stop sweeping once we OOM on prefill (longer will also OOM)
        if info["status"] == "oom":
            print("  (prefill OOM; stopping ladder for this method)")
            break

    # summary
    print("\n" + "=" * 80 + "\nSUMMARY (NIAH length ladder)")
    print(f"{'ctx':>7} {'tok':>7} {'status':>8} {'peak_gb':>8} {'acc':>5}")
    for r in rows:
        acc_str = "-" if r["acc"] is None else f"{r['acc']:.2f}"
        peak = r["prefill_peak_gb"] or 0.0
        print(f"{r['ctx_len']:>7} {r['true_len']:>7} {r['prefill_status']:>8} "
              f"{peak:>8.2f} {acc_str:>5}")

    # derived headline summary (the numbers paper.tex needs)
    ok_rows = [r for r in rows if r["prefill_status"] == "ok"]
    oom_rows = [r for r in rows if r["prefill_status"] == "oom"]
    best = max(ok_rows, key=lambda r: r["true_len"], default=None)
    summary = {
        "max_ok_ctx_len": (best["ctx_len"] if best else None),
        "max_ok_true_len": (best["true_len"] if best else None),
        "peak_gb_at_max_ok": (best["prefill_peak_gb"] if best else None),
        "acc_at_max_ok": (best["acc"] if best else None),
        "n_ok": len(ok_rows), "n_oom": len(oom_rows),
        "first_oom_ctx_len": (oom_rows[0]["ctx_len"] if oom_rows else None),
        "mean_acc_ok": (round(sum((r["acc"] or 0) for r in ok_rows) / len(ok_rows), 4)
                        if ok_rows else None),
    }
    print(f"\nHEADLINE: max OOM-free ctx_len={summary['max_ok_ctx_len']} "
          f"(tok={summary['max_ok_true_len']}) peak={summary['peak_gb_at_max_ok']}GB "
          f"acc={summary['acc_at_max_ok']} | first_oom={summary['first_oom_ctx_len']}")

    record = {
        "experiment": "niah", "timestamp": env["timestamp"],
        "env": env, "args": vars(a), "cfg": cfg.__dict__,
        "summary": summary, "rows": rows,
    }
    save_run("niah", record)


if __name__ == "__main__":
    main()
