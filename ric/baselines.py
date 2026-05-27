# ------------------------------------------------------------------------------
# Prefill strategies under comparison. All return a cache object ready for the
# answer phase; the experiment scripts measure PEAK memory around these calls.
#
#   full      : prefill the whole context, no compression. Peak = full KV.
#   kvzip     : KVzip -- full prefill + reconstruction scoring + per-head prune.
#               Peak >= full KV (scoring adds overhead); compression only helps
#               AFTER the peak. This is the baseline our thesis targets.
#   streamllm : streaming prefill but evict by RECENCY (no scoring). Bounds peak
#               like us, but drops the middle of the context -> weak accuracy.
#   ours      : streaming prefill with reconstruction-based compaction
#               (--per_token / --per_head). Bounds peak AND keeps important KV.
# ------------------------------------------------------------------------------
import torch

import ric  # noqa: F401

from .stream_prefill import RICConfig, streaming_prefill


@torch.inference_mode()
def prefill_full(model, ctx_ids, chunk=16000):
    # do_score=False: skip scoring; no prune -> full dense KV retained.
    # prefill_chunk_size matched to ours so only the STORED KV differs (fair peak).
    return model.prefill(ctx_ids, prefill_chunk_size=chunk, load_score=False, do_score=False)


@torch.inference_mode()
def prefill_kvzip(model, ctx_ids, ratio=0.3, chunk=16000):
    kv = model.prefill(ctx_ids, prefill_chunk_size=chunk, load_score=False, do_score=True)
    kv.prune(ratio, level="pair")                                  # per-head varlen eviction
    return kv


@torch.inference_mode()
def prefill_streamllm(model, ctx_ids, cfg: RICConfig, finalize=True):
    cfg2 = RICConfig(**{**cfg.__dict__, "importance": "recent", "level": "per_token"})
    return streaming_prefill(model, ctx_ids, cfg2, finalize=finalize)


@torch.inference_mode()
def prefill_ours(model, ctx_ids, cfg: RICConfig, finalize=True):
    cfg2 = RICConfig(**{**cfg.__dict__, "importance": "score"})
    return streaming_prefill(model, ctx_ids, cfg2, finalize=finalize)


def build_prefill_fn(method: str, model, cfg: RICConfig, ratio: float = 0.3, finalize=True):
    """Return a zero-arg-ish callable `fn(ctx_ids) -> kv` for the chosen method,
    so the experiment harness can wrap it in peak-memory measurement."""
    if method == "full":
        return lambda ctx: prefill_full(model, ctx, chunk=cfg.chunk)
    if method == "kvzip":
        return lambda ctx: prefill_kvzip(model, ctx, ratio=ratio, chunk=cfg.chunk)
    if method == "streamllm":
        return lambda ctx: prefill_streamllm(model, ctx, cfg, finalize=finalize)
    if method in ("ours", "ric"):
        return lambda ctx: prefill_ours(model, ctx, cfg, finalize=finalize)
    raise ValueError(f"unknown method: {method}")
