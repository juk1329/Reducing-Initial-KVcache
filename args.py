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
    p.add_argument("--budget_max", type=int, default=8192, help="compaction trigger")
    p.add_argument("--budget_target", type=int, default=4096, help="compact down to this")
    p.add_argument("--recent_window", type=int, default=256,
                   help="freshest tokens never evicted during prefill")
    p.add_argument("--chunk", type=int, default=2048, help="prefill+scoring chunk size")
    p.add_argument("--token_agg", default="mean", choices=["mean", "max"])
    p.add_argument("--rescore", action="store_true",
                   help="re-score cached context at each compaction (user's variant)")
    p.add_argument("--head_ratio", type=float, default=0.5,
                   help="per_head finalize: retained fraction of the compacted context")
    # kvzip baseline
    p.add_argument("--ratio", type=float, default=0.3, help="kvzip prune ratio (retained)")
    # generation / misc
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--tag", default=None)
    return p


def cfg_from_args(a) -> RICConfig:
    return RICConfig(
        level=a.level,
        budget_max=a.budget_max,
        budget_target=a.budget_target,
        recent_window=a.recent_window,
        chunk=a.chunk,
        token_agg=a.token_agg,
        importance="score",
        rescore=a.rescore,
        head_ratio=a.head_ratio,
    )
