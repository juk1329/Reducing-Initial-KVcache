#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Experiment 2 - SCBench short-context multi-turn: "with a (relatively) short
# shared context and an accumulating multi-turn session, how many turns/tokens
# can each method sustain before OOM, and how well does accuracy hold up?"
#
# The shared context is prefilled once (small for the *_tiny tags ~8k tokens),
# then we run a multi-turn dialogue with cache accumulation (update_cache=True).
# Our method bounds the accumulating KV (compacted context + recent turns), so it
# survives long sessions; `full`/`kvzip` accumulate without bound and OOM sooner.
#
# Does NOT auto-run. Example:
#   conda run -n jk python run_scbench.py -m llama3.2-3b -d scbench_kv_tiny \
#       --method ours --turns 200 --turn_budget 2048
# ------------------------------------------------------------------------------
import argparse
import gc
import json
import os
import re
import string

import torch

from args import add_common_args, cfg_from_args
import ric
from ric.baselines import build_prefill_fn
from ric.mem import run_oom_safe, total_gb, reset_peak, peak_gb
from ric.record import capture_env, save_run
from ric.stream_prefill import answer, evict_old_turns
from ric.stream_cache import StreamingCache

from model import ModelKVzip
from data.load import load_dataset_all


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_answer(pred: str, gold: str) -> float:
    """Loose containment score in [0,1]: 1 if the gold answer appears in the
    prediction (normalized), else token-overlap fraction."""
    p, g = normalize(pred), normalize(gold)
    if not g:
        return 0.0
    if g in p:
        return 1.0
    gt = set(g.split())
    pt = set(p.split())
    return len(gt & pt) / max(1, len(gt))


def main():
    pa = argparse.ArgumentParser()
    add_common_args(pa)
    pa.add_argument("-d", "--data", default="scbench_kv_tiny",
                    help="SCBench task (e.g. scbench_kv_tiny, scbench_qa_eng, scbench_mf_tiny)")
    pa.add_argument("--idx", type=int, default=0, help="dataset example index")
    pa.add_argument("--turns", type=int, default=0,
                    help="number of turns (cycles the example's questions; 0 = all once)")
    pa.add_argument("--turn_budget", type=int, default=2048,
                    help="ours: max accumulated turn tokens kept (recent)")
    a = pa.parse_args()
    cfg = cfg_from_args(a)

    # SCBench multi-turn accumulation needs a DENSE cache for turn eviction, so we
    # run the per_token (dense) finalize even if --level per_head was passed.
    if a.method == "ours" and a.level == "per_head":
        print("[note] per_head multi-turn (varlen) is future work; using per_token "
              "(dense) compaction for this SCBench run.")
        cfg.level = "per_token"

    print(f"GPU total ~{total_gb(a.device):.1f}GB | model={a.model} data={a.data} "
          f"method={a.method} budget_max={a.budget_max} target={a.budget_target} "
          f"turn_budget={a.turn_budget}")
    model = ModelKVzip(a.model, kv_type="evict")
    model.set_chat_template(a.data)
    env = capture_env(model, a.device)
    print(f"env: {env.get('gpu_name')} {env.get('gpu_total_gb')}GB | "
          f"KV {env.get('kv_gb_per_1k_tokens')} GB/1k tok | torch {env.get('torch')} "
          f"tf {env.get('transformers')} fa {env.get('flash_attn')}")

    dataset = load_dataset_all(a.data, model.tokenizer)
    ex = dataset[a.idx]
    ctx_ids = model.encode(ex["context"])
    questions = list(ex["question"])
    golds = list(ex["answers"]) if ex.get("answers") else [""] * len(questions)
    n_turns = a.turns if a.turns > 0 else len(questions)
    print(f"context tokens={ctx_ids.shape[1]} | #questions={len(questions)} | turns={n_turns}")

    # ---- prefill the shared context (measure peak) ----
    prefill_fn = build_prefill_fn(a.method, model, cfg, ratio=a.ratio)
    kv, info = run_oom_safe(lambda: prefill_fn(ctx_ids), device=a.device)
    if info["status"] != "ok":
        print(f"[abort] context prefill {info['status']} at peak~{info['peak_gb']}")
        return
    print(f"context prefilled: peak={info['peak_gb']:.2f}GB cache={kv._mem():.2f}GB")

    is_stream = isinstance(kv, StreamingCache)
    base_phys = kv.phys_total() if is_stream else None
    prefill_stats = {
        "ctx_tokens": int(ctx_ids.shape[1]),
        "peak_gb": info["peak_gb"], "cache_gb": float(kv._mem()),
        "num_compactions": getattr(kv, "num_compactions", None),
        "peak_phys_ctx": getattr(kv, "peak_phys_ctx", None),
    }

    # ---- multi-turn dialogue with accumulation ----
    turns = []
    survived = 0
    reset_peak(a.device)
    for t in range(n_turns):
        q = questions[t % len(questions)]
        gold = golds[t % len(golds)] if golds else ""

        res, tinfo = run_oom_safe(
            lambda: answer(model, kv, q, max_new=a.max_new, update_cache=True),
            device=a.device,
        )
        if tinfo["status"] == "oom":
            print(f"  turn {t}: OOM (survived {survived} turns)")
            break

        if is_stream and a.turn_budget:
            evict_old_turns(kv, base_phys, a.turn_budget)

        acc = score_answer(res, gold)
        rec = {"turn": t, "cache_gb": float(kv._mem()),
               "phys": int(kv.phys_total()) if is_stream else int(kv.get_seq_length()),
               "acc": acc, "answer": res[:100], "gold": str(gold)[:100]}
        turns.append(rec)
        survived += 1
        if t % 10 == 0 or t == n_turns - 1:
            print(f"  turn {t}: cache={rec['cache_gb']:.2f}GB phys={rec['phys']} "
                  f"acc={acc:.2f} | {res[:60]!r}", flush=True)

    accs = [r["acc"] for r in turns]
    mean_acc = sum(accs) / len(accs) if accs else 0.0
    peak_turns = peak_gb(a.device)
    print("\n" + "=" * 80)
    print(f"SUMMARY: method={a.method} survived_turns={survived}/{n_turns} "
          f"mean_acc={mean_acc:.3f} final_cache={(turns[-1]['cache_gb'] if turns else 0):.2f}GB "
          f"peak_during_turns={peak_turns:.2f}GB")

    summary = {
        "method": a.method, "level": cfg.level,
        "n_turns_requested": n_turns, "survived_turns": survived,
        "completed_all": survived >= n_turns,
        "mean_acc": round(mean_acc, 4),
        "ctx_tokens": prefill_stats["ctx_tokens"],
        "prefill_peak_gb": info["peak_gb"],
        "peak_during_turns_gb": peak_turns,
        "final_cache_gb": (turns[-1]["cache_gb"] if turns else None),
    }
    record = {
        "experiment": "scbench", "timestamp": env["timestamp"],
        "env": env, "args": vars(a), "cfg": cfg.__dict__,
        "prefill": prefill_stats, "summary": summary, "turns": turns,
    }
    save_run("scbench", record)

    del kv
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
