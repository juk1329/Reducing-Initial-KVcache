# ------------------------------------------------------------------------------
# Shared CLI for the experiment scripts.
# ------------------------------------------------------------------------------
import argparse

from ric.stream_prefill import RICConfig


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("-m", "--model", default="llama3.2-3b",
                   help="KVzip model name (e.g. llama3.2-1b, llama3.2-3b, qwen3-1.7b)")
    p.add_argument("--method", default="ours",
                   choices=["full", "kvzip", "streamllm", "ours"],
                   help="prefill strategy under test")
    p.add_argument("--level", default="per_token", choices=["per_token", "per_head"],
                   help="ours: token-level (dense) or per-head (KVzip varlen) finalize")
    # streaming compaction budgets (in context tokens)
    p.add_argument("--intermediate_budget", type=int, default=2048,
                   help="B_i: tokens KEPT per committed segment (heavy-prune target)")
    p.add_argument("--working_max", type=int, default=0,
                   help="heavy-prune trigger: working window size (0 -> 2*intermediate_budget)")
    p.add_argument("--final_budget", type=int, default=8192,
                   help="B_f: cap on total committed context (multiple of intermediate_budget)")
    p.add_argument("--recent_window", type=int, default=256,
                   help="freshest tokens never evicted during a prune")
    p.add_argument("--chunk", type=int, default=2048, help="prefill+scoring chunk size")
    p.add_argument("--token_agg", default="mean", choices=["mean", "max"])
    p.add_argument("--light_drop_ratio", type=float, default=0.05,
                   help="Hybrid per-chunk light drop: drop score < ratio*chunk_max (0 disables)")
    p.add_argument("--no_rescore_working", action="store_true",
                   help="disable re-scoring the working window before each heavy prune "
                        "(reuse fresh-chunk scores instead)")
    p.add_argument("--head_ratio", type=float, default=0.5,
                   help="per_head finalize: retained fraction of the compacted context")
    # predict-prompt variant: intermediate (chunk/segment) prunes score with a PREDICT
    # prompt instead of the repeat prompt; the final compaction always uses repeat.
    p.add_argument("--use_predict_prompt", action="store_true",
                   help="use a predict prompt for intermediate peak-reducing prunes "
                        "(final compaction stays on the repeat prompt)")
    p.add_argument("--predict_prompt_version", type=int, default=1, choices=[1, 2],
                   help="1: 'Predict the entire context:'  2: 'Predict the upcoming context:'")
    p.add_argument("--predict_target", default="self", choices=["self", "next_chunk"],
                   help="self: predict prompt scores a chunk on itself; next_chunk: score a "
                        "chunk by the REAL next chunk's attention (keeps first/last chunk as "
                        "sink/recent, predict-scores only the middle)")
    p.add_argument("--combine_repeat", action="store_true",
                   help="intermediate scoring = combine(repeat, predict) per position "
                        "(recovers repeat's retrieval signal while keeping predict's)")
    p.add_argument("--combine_mode", default="max", choices=["max", "wsum"],
                   help="how to merge repeat & predict scores")
    p.add_argument("--combine_alpha", type=float, default=0.5,
                   help="wsum weight on the predict signal: alpha*predict+(1-alpha)*repeat")
    p.add_argument("--sink_tokens", type=int, default=4,
                   help="predict-next boundary protection: # leading context tokens kept as an "
                        "attention sink (StreamingLLM-style, small; recent tail uses recent_window)")
    # kvzip baseline
    p.add_argument("--ratio", type=float, default=0.3, help="kvzip prune ratio (retained)")
    # compression-ratio sweep: a single RETAINED fraction r applied identically to both
    # methods. kvzip -> prune ratio=r; ours -> final_budget=round(r*ctx_len) with B_i/
    # working_max derived per length (see derive_ric_budgets). <0 disables (use explicit
    # budgets / --ratio instead).
    p.add_argument("--comp_ratio", type=float, default=-1.0,
                   help="retained fraction r in [0,1] applied to both ours and kvzip (<0 = off)")
    # generation / misc
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--tag", default=None)
    return p


def cfg_from_args(a) -> RICConfig:
    working_max = a.working_max if a.working_max > 0 else 2 * a.intermediate_budget
    return RICConfig(
        level=a.level,
        intermediate_budget=a.intermediate_budget,
        working_max=working_max,
        final_budget=a.final_budget,
        recent_window=a.recent_window,
        chunk=a.chunk,
        token_agg=a.token_agg,
        light_drop_ratio=a.light_drop_ratio,
        importance="score",
        rescore_working=not a.no_rescore_working,
        head_ratio=a.head_ratio,
        use_predict_prompt=a.use_predict_prompt,
        predict_prompt_version=a.predict_prompt_version,
        predict_target=a.predict_target,
        combine_repeat=a.combine_repeat,
        combine_mode=a.combine_mode,
        combine_alpha=a.combine_alpha,
        sink_tokens=a.sink_tokens,
    )
