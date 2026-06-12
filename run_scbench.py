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

from dataclasses import replace

from args import add_common_args, cfg_from_args
import ric
from ric.baselines import build_prefill_fn
from ric.mem import run_oom_safe, total_gb, reset_peak, peak_gb
from ric.record import capture_env, save_run
from ric.stream_prefill import (answer, evict_old_turns, multiturn_compress,
                                derive_ric_budgets)
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
                    help="streamllm/baseline: max accumulated turn tokens kept (recent FIFO)")
    pa.add_argument("--consec_zero_stop", type=int, default=5,
                    help="stop (record, don't reach OOM) after this many consecutive 0-acc turns")
    pa.add_argument("--turn_base_budget", type=int, default=-1,
                    help="ours: base budget for turn compression (B_turn=0.1*this, fixed "
                         "cap=turn_cap_mult*this). <=0 uses the context final_budget B_f. Set "
                         "small (e.g. 2048) when the context is uncompressed (comp_ratio 1.0).")
    pa.add_argument("--turn_ratio", type=float, default=0.4,
                    help="ours: DYNAMIC per-turn compression — keep round(turn_ratio*turn_len) of "
                         "each turn by importance (whole session compressed at this ratio). "
                         "<0 disables -> legacy FIXED-cap mode.")
    pa.add_argument("--turn_cap_mult", type=int, default=2,
                    help="ours LEGACY (turn_ratio<0): FIXED multiple for the committed turn-region "
                         "cap (cap = turn_cap_mult * turn_base).")
    pa.add_argument("--turn_rescore_every", type=int, default=8,
                    help="ours: do the expensive full REPEAT re-score of the turn region every "
                         "N cap-reprunes (cheap top-k reprune still runs every turn).")
    a = pa.parse_args()
    cfg = cfg_from_args(a)

    # SCBench multi-turn needs a DENSE cache (dense turn append + streaming compaction), so
    # ours runs the per_token finalize even if --level per_head was passed.
    if a.method == "ours" and a.level == "per_head":
        print("[note] per_head multi-turn (varlen) is future work; using per_token "
              "(dense) compaction for this SCBench run.")
        cfg.level = "per_token"

    # method variant label (ours_combine vs ours_predict vs ours) for recording/aggregation
    method_variant = a.method
    if a.method == "ours":
        method_variant = ("ours_combine" if a.combine_repeat
                          else f"ours_predict_v{a.predict_prompt_version}" if a.use_predict_prompt
                          else "ours")

    print(f"GPU total ~{total_gb(a.device):.1f}GB | model={a.model} data={a.data} "
          f"method={method_variant} comp_ratio={a.comp_ratio} turn_budget={a.turn_budget}")
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
    ctx_len = int(ctx_ids.shape[1])

    # ---- fair context compression: a shared retained ratio for both methods ----
    # ours -> final_budget = round(r*ctx_len) (per_token streaming budgets derived);
    # kvzip -> prune ratio = r. (<0 disables -> use explicit --final_budget / --ratio.)
    use_cr = a.comp_ratio is not None and a.comp_ratio >= 0.0
    eff_ratio = a.comp_ratio if use_cr else a.ratio
    if use_cr and a.method == "ours":
        fb, bi, wm = derive_ric_budgets(a.comp_ratio, ctx_len, cfg.chunk, cfg.recent_window)
        cfg = replace(cfg, final_budget=fb, intermediate_budget=bi, working_max=wm)
    B_f = cfg.final_budget
    print(f"context tokens={ctx_len} | #questions={len(questions)} | turns={n_turns} | "
          f"B_f={B_f} B_turn={max(1, round(0.1*B_f))} (ours) | kvzip ratio={eff_ratio}")

    # ---- prefill the shared context (measure peak) ----
    prefill_fn = build_prefill_fn(a.method, model, cfg, ratio=eff_ratio)
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
        "num_segments": getattr(kv, "num_segments", None),
        "peak_phys_ctx": getattr(kv, "peak_phys_ctx", None),
    }

    # ---- multi-turn dialogue ----
    if a.method == "ours":
        # RIC streaming turn-compression: each (query+answer) is a chunk, compressed to
        # 0.1*B_f; hard repeat re-compress when committed turns exceed m*B_f (m grows).
        turn_base = a.turn_base_budget if a.turn_base_budget > 0 else B_f
        mt = multiturn_compress(model, kv, questions, golds, cfg, score_answer, turn_base,
                                max_new=a.max_new, max_turns=n_turns,
                                consec_zero_stop=a.consec_zero_stop, turn_ratio=a.turn_ratio,
                                turn_cap_mult=a.turn_cap_mult, rescore_every=a.turn_rescore_every,
                                device=a.device, verbose=True)
        turns = mt["turns"]
        survived = mt["survived_turns"]
        stop_reason = mt["stop_reason"]
        peak_turns = mt["peak_during_turns_gb"]
        mean_acc = mt["mean_acc"] or 0.0
        max_tokens = mt["max_logical_tokens"]
        final_cache = mt["final_cache_gb"]
        turn_cap = mt["turn_cap"]
        turn_ratio_eff = mt["turn_ratio"]
        n_turn_reprunes = mt["n_turn_reprunes"]
    else:
        # baseline (kvzip / full / streamllm): turns accumulate. streamllm bounds via recent
        # FIFO; kvzip/full accumulate without bound -> OOM is the headline. Same stop rules.
        turns, survived, consec_zero, stop_reason, max_tokens = [], 0, 0, "completed", ctx_len
        turn_cap = None
        turn_ratio_eff = None
        n_turn_reprunes = None
        reset_peak(a.device)
        for t in range(n_turns):
            q = questions[t % len(questions)]
            gold = golds[t % len(golds)] if golds else ""
            res, tinfo = run_oom_safe(
                lambda: answer(model, kv, q, max_new=a.max_new, update_cache=True),
                device=a.device,
            )
            if tinfo["status"] != "ok":
                stop_reason = tinfo["status"]  # 'oom' / 'error'
                print(f"  turn {t}: {stop_reason.upper()} (survived {survived} turns, ~{max_tokens} tok)")
                break
            if is_stream and a.turn_budget:
                evict_old_turns(kv, base_phys, a.turn_budget)
            tok = int(kv.get_seq_length())
            max_tokens = max(max_tokens, tok)
            acc = score_answer(res, gold)
            consec_zero = consec_zero + 1 if acc == 0.0 else 0
            rec = {"turn": t, "cache_gb": float(kv._mem()),
                   "phys": int(kv.phys_total()) if is_stream else tok,
                   "logical_tokens": tok, "committed_turns": None, "turn_cap_mult": None,
                   "acc": acc, "answer": res[:120], "gold": str(gold)[:80]}
            turns.append(rec)
            survived += 1
            if t % 10 == 0 or t == n_turns - 1:
                print(f"  turn {t}: cache={rec['cache_gb']:.2f}GB phys={rec['phys']} "
                      f"acc={acc:.2f} | {res[:50]!r}", flush=True)
            if consec_zero >= a.consec_zero_stop:
                stop_reason = "acc_collapse"
                print(f"  turn {t}: acc collapse ({consec_zero} consecutive 0s) -> stop")
                break
        accs = [r["acc"] for r in turns]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        peak_turns = peak_gb(a.device)
        final_cache = (turns[-1]["cache_gb"] if turns else None)

    print("\n" + "=" * 80)
    print(f"SUMMARY: method={method_variant} survived_turns={survived}/{n_turns} "
          f"stop={stop_reason} max_tokens={max_tokens} mean_acc={mean_acc:.3f} "
          f"final_cache={(final_cache or 0):.2f}GB peak_during_turns={peak_turns:.2f}GB")

    summary = {
        "method": a.method, "method_variant": method_variant, "level": cfg.level,
        "comp_ratio": (a.comp_ratio if use_cr else None), "B_f": B_f,
        "n_turns_requested": n_turns, "survived_turns": survived,
        "completed_all": survived >= n_turns, "stop_reason": stop_reason,
        "max_tokens_processed": max_tokens,
        "turn_cap": turn_cap, "turn_ratio": turn_ratio_eff, "n_turn_reprunes": n_turn_reprunes,
        "mean_acc": round(mean_acc, 4),
        "ctx_tokens": prefill_stats["ctx_tokens"],
        "prefill_peak_gb": info["peak_gb"],
        "peak_during_turns_gb": peak_turns,
        "final_cache_gb": final_cache,
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
