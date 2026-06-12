#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Experiment 1b - NIAH compression-ratio GRID (per-head, fixed small contexts).
#
# Long-context / OOM behaviour now lives in SCBench. Here we use small fixed
# contexts (500 / 2000 / 8000 tokens) and, at each (ctx_len, ratio), measure:
#   (a) PREFILL PEAK memory -- the headline: how much smaller is ours' peak than
#       kvzip's at the SAME overall retained ratio (no OOM at these sizes)?
#   (b) needle accuracy at 11 quantile positions (depth 0,10,..,100).
#
# Fairness fixes vs the old ladder script:
#   * The NIAH context for a given (ctx_len, depth) is generated ONCE and the
#     identical token ids are fed to every (ratio, method) cell. (The generator
#     returns text and re-encoding is lossy at BPE boundaries, so separate runs
#     used to get different true_len -> not comparable.)
#   * All methods run with level=per_head (kvzip is inherently per-head).
#
# Methods (all per-head):
#   kvzip            : full prefill + per-head prune to ratio r.
#   ours             : streaming compaction, repeat prompt throughout.
#   ours_predict_v1  : streaming, PREDICT prompt ("Predict the entire context:")
#                      for intermediate prunes, repeat for the final compaction.
#   ours_predict_v2  : same, predict prompt v2 ("Predict the upcoming context:").
#
#   conda run -n jk python run_niah_grid.py -m qwen3-4b
# ------------------------------------------------------------------------------
import argparse
import gc
import time

import torch
from dataclasses import replace

from args import add_common_args, cfg_from_args
import ric  # noqa: F401
from ric.baselines import prefill_kvzip
from ric.mem import run_oom_safe, total_gb
from ric.record import capture_env, save_run
from ric.stream_prefill import answer, derive_ric_budgets_for_ratio, streaming_prefill
from model import ModelKVzip
from run_niah import make_niah, score_answer

# grid method name -> (base, use_predict_prompt, predict_prompt_version, predict_target, combine_repeat)
METHOD_SPECS = {
    "kvzip":             ("kvzip", False, 1, "self",       False),
    "ours":              ("ours",  False, 1, "self",       False),
    "ours_predict_v1":   ("ours",  True,  1, "self",       False),
    "ours_predict_v2":   ("ours",  True,  2, "self",       False),
    "ours_predict_next": ("ours",  True,  1, "next_chunk", False),  # predict target = real next chunk
    "ours_combine":      ("ours",  True,  1, "next_chunk", True),   # combine(repeat, predict-next)
}


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--ctx_lens", default="500,2000,8000",
                   help="comma-separated fixed context lengths")
    p.add_argument("--depths", default="0,10,20,30,40,50,60,70,80,90,100",
                   help="needle insertion positions in percent (quantiles x100)")
    p.add_argument("--ratios", default="0.0,0.2,0.4,0.6,0.8,1.0",
                   help="overall retained fractions applied to BOTH methods")
    p.add_argument("--methods", default="kvzip,ours,ours_predict_v1,ours_predict_v2")
    p.add_argument("--head_ratio_target", type=float, default=0.5,
                   help="ours per-head target retention; positional fraction = min(1, r/target)")
    # small contexts -> small streaming granularity so ours actually segments at 2k/8k
    p.set_defaults(chunk=512, recent_window=64, level="per_head")
    a = p.parse_args()

    ctx_lens = [int(x) for x in a.ctx_lens.split(",")]
    depths = [int(x) for x in a.depths.split(",")]
    ratios = [float(x) for x in a.ratios.split(",")]
    methods = [m.strip() for m in a.methods.split(",")]
    for m in methods:
        if m not in METHOD_SPECS:
            raise ValueError(f"unknown method {m!r}; choose from {list(METHOD_SPECS)}")
    cfg = cfg_from_args(a)  # level=per_head, importance=score

    print(f"GPU total ~{total_gb(a.device):.1f}GB | model={a.model} level=per_head "
          f"chunk={a.chunk} recent_window={a.recent_window} head_ratio_target={a.head_ratio_target}")
    print(f"grid: ctx={ctx_lens} depths={depths} ratios={ratios} methods={methods} "
          f"-> {len(ctx_lens)*len(depths)*len(ratios)*len(methods)} cells")
    model = ModelKVzip(a.model, kv_type="evict")
    env = capture_env(model, a.device)
    print(f"env: {env.get('gpu_name')} {env.get('gpu_total_gb')}GB | "
          f"KV {env.get('kv_gb_per_1k_tokens')} GB/1k tok | torch {env.get('torch')}")

    rows = []
    for L in ctx_lens:
        for depth in depths:
            # generate the NIAH context ONCE; reuse the SAME ids for every method/ratio
            ctx, q, gold = make_niah(model, L, depth)
            ctx_ids = model.encode(ctx)
            true_len = int(ctx_ids.shape[1])
            quantile = round(depth / 100.0, 2)
            print("=" * 80,
                  f"\n[ctx_len={L} depth={depth}% q={quantile} true_len={true_len}]", flush=True)

            for r in ratios:
                for mname in methods:
                    base, use_pred, pver, ptarget, comb = METHOD_SPECS[mname]
                    if base == "kvzip":
                        prefill_fn = lambda r=r: prefill_kvzip(model, ctx_ids, ratio=r, chunk=a.chunk)
                        fb = bi = wm = None
                        hr_eff = r  # kvzip retains fraction r (per-head)
                    else:
                        fb, bi, wm, hr_eff, rw = derive_ric_budgets_for_ratio(
                            r, true_len, a.chunk, a.recent_window, a.head_ratio_target)
                        cfg_L = replace(cfg, final_budget=fb, intermediate_budget=bi,
                                        working_max=wm, head_ratio=hr_eff, recent_window=rw,
                                        level="per_head", use_predict_prompt=use_pred,
                                        predict_prompt_version=pver, predict_target=ptarget,
                                        combine_repeat=comb)
                        prefill_fn = lambda c=cfg_L: streaming_prefill(
                            model, ctx_ids, c, finalize=True, verbose=False)

                    t0 = time.time()
                    kv, info = run_oom_safe(prefill_fn, device=a.device)
                    row = {
                        "model": a.model, "method": mname, "base_method": base,
                        "ctx_len": L, "true_len": true_len, "depth": depth,
                        "needle_quantile": quantile, "comp_ratio": r, "level": "per_head",
                        "use_predict_prompt": use_pred, "predict_prompt_version": pver,
                        "predict_target": ptarget, "combine_repeat": comb,
                        "head_ratio_eff": round(float(hr_eff), 4),
                        "final_budget": fb, "intermediate_budget": bi, "working_max": wm,
                        "num_segments": None, "cache_gb": None,
                        "prefill_status": info["status"], "prefill_peak_gb": info["peak_gb"],
                        "prefill_sec": round(time.time() - t0, 2),
                        "acc": None, "answer": None, "question": q, "gold": gold,
                        "error": info.get("error"),
                    }
                    if info["status"] == "ok":
                        row["num_segments"] = getattr(kv, "num_segments", None)
                        try:
                            row["cache_gb"] = float(kv._mem())
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            ans = answer(model, kv, q, max_new=a.max_new, update_cache=True)
                            row["acc"] = score_answer(ans)
                            row["answer"] = ans[:200]
                        except Exception as e:  # noqa: BLE001
                            row["answer"] = f"ANSWER_ERROR: {str(e)[:200]}"
                    rows.append(row)
                    peak = row["prefill_peak_gb"] or 0.0
                    acc_s = "-" if row["acc"] is None else f"{row['acc']:.2f}"
                    print(f"  r={r:<3} {mname:16s} peak={peak:6.3f}GB acc={acc_s:>4} "
                          f"hr={row['head_ratio_eff']} fb={fb} segs={row['num_segments']}",
                          flush=True)
                    del kv
                    gc.collect()
                    torch.cuda.empty_cache()

    record = {
        "experiment": "niah_grid", "timestamp": env["timestamp"],
        "env": env, "args": vars(a), "cfg": cfg.__dict__, "rows": rows,
    }
    save_run("niah_grid", record)
    print(f"\nDONE: {len(rows)} cells recorded.")


if __name__ == "__main__":
    main()
