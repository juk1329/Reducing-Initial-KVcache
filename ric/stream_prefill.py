# ------------------------------------------------------------------------------
# Streaming prefill driver (Method B).
#
# For each context chunk we:
#   (1) prefill it into the cache (dense append),
#   (2) score it with KVzip's reconstruction task ("repeat this chunk") -- KVzip's
#       scoring is chunk-local, so a fresh chunk can be scored immediately without
#       the rest of the context,
#   (3) if the physical context exceeds `budget_max`, compact down to
#       `budget_target` (token-level), optionally re-scoring first (`--rescore`).
#
# This bounds the PEAK physically-stored context KV at ~budget_max + chunk, which
# is the whole point: baselines must materialize the full-context KV first.
# ------------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import List, Optional

import torch

import ric  # noqa: F401
from model.wrapper import chunk_fn

from .stream_cache import StreamingCache


@dataclass
class RICConfig:
    level: str = "per_token"          # per_token | per_head
    budget_max: int = 8192            # compaction trigger (physical context tokens)
    budget_target: int = 4096         # compact down to this many context tokens
    recent_window: int = 256          # most-recent tokens never evicted during prefill
    chunk: int = 2048                 # prefill + scoring chunk size
    token_agg: str = "mean"           # mean | max  (collapse head/layer scores -> per position)
    importance: str = "score"         # score (reconstruction) | recent (StreamingLLM baseline)
    rescore: bool = False             # re-score the cached context at each compaction
    head_ratio: float = 0.5           # per_head finalize: retained fraction after streaming
    prev_tail: int = 8                # hint tokens carried from the previous chunk


def build_repeat(model, a_ids: torch.Tensor, prev_tail: Optional[torch.Tensor]):
    """Mirror KVzip's self_task: build the reconstruction ('repeat') query for a
    chunk. `a_ids` are the chunk's tokens (the reconstruction target)."""
    if prev_tail is None:
        q = model.encode("\n\nRepeat the previous context exactly.")
    else:
        q = model.encode("\n\nRepeat the part of the previous context exactly, starting with ")
        q = torch.cat([q, prev_tail], dim=1)
    return torch.cat([q, model.postfix_ids, a_ids], dim=1)


@torch.inference_mode()
def _score_fresh_chunk(model, kv: StreamingCache, chunk: torch.Tensor,
                       prev_tail: Optional[torch.Tensor]):
    """Run the reconstruction forward for the just-prefilled chunk and append its
    per-position importance scores to kv.score."""
    base = model.model.model  # HF base model (e.g. LlamaModel); avoids ModelKVzip.__call__'s slice
    c = chunk.shape[-1]
    start, end = kv.score_region_for_fresh_chunk(c)
    kv.start_idx, kv.end_idx = start, end
    repeat_ids = build_repeat(model, chunk, prev_tail)
    kv.mark()
    kv.get_score = True
    base(repeat_ids, past_key_values=kv)
    kv.get_score = False
    kv.restore()


@torch.inference_mode()
def rescore_cache(model, kv: StreamingCache, cfg: RICConfig):
    """Optional (`--rescore`): re-evaluate importance of the CURRENTLY cached
    context by re-running reconstruction over the surviving tokens, before
    culling. This realizes the user's 'low threshold early, re-run reconstruct
    prompt and cull harder when budget is hit' variant.

    Caveat: the surviving keys keep their original (gappy) RoPE phase, so this is
    an approximation -- the same one KVzip's post-prune inference relies on."""
    base = model.model.model
    ids = kv.ctx_token_ids  # [1, P]
    kv.init_score()
    kv.get_score = False
    prev_tail = None
    pos = 0
    for w in chunk_fn(ids, cfg.chunk):
        c = w.shape[-1]
        kv.start_idx = kv.sink + pos
        kv.end_idx = kv.sink + pos + c
        repeat_ids = build_repeat(model, w, prev_tail)
        kv.mark()
        kv.get_score = True
        base(repeat_ids, past_key_values=kv)
        kv.get_score = False
        kv.restore()
        prev_tail = w[:, -cfg.prev_tail:]
        pos += c
    assert kv.score[0].shape[-1] == kv.ctx_len


@torch.inference_mode()
def streaming_prefill(model, ctx_ids: torch.Tensor, cfg: RICConfig,
                      finalize: bool = True, verbose: bool = True) -> StreamingCache:
    """Prefill `ctx_ids` with streaming compaction. Returns a StreamingCache ready
    for query answering. `model` is a KVzip ModelKVzip instance."""
    ctx_ids = ctx_ids.to(model.device)
    sys_ids = model.sys_prompt_ids
    sink = sys_ids.shape[1]

    kv = StreamingCache(
        model.model, sink_len=sink,
        budget_max=cfg.budget_max, budget_target=cfg.budget_target,
        recent_window=cfg.recent_window, token_agg=cfg.token_agg,
        importance=cfg.importance,
    )

    # 0) prefill the system prompt (the protected sink region)
    model(sys_ids, kv, update_cache=True)
    kv.init_score()
    kv.get_score = False
    kv.pruned = False

    # 1) stream the context
    prev_tail = None
    for ci, chunk in enumerate(chunk_fn(ctx_ids, cfg.chunk)):
        model(chunk, kv, update_cache=True)         # prefill (dense append)
        kv.note_chunk_prefilled(chunk)
        if cfg.importance == "score":
            _score_fresh_chunk(model, kv, chunk, prev_tail)
        prev_tail = chunk[:, -cfg.prev_tail:]
        if kv.need_compact():
            if cfg.rescore and cfg.importance == "score":
                rescore_cache(model, kv, cfg)
            kv.compact()
        if verbose:
            print(f"  chunk {ci}: logical_ctx={kv.logical_ctx_len} "
                  f"phys_ctx={kv.ctx_len} cache={kv.mem_gb():.2f}GB "
                  f"compactions={kv.num_compactions}", flush=True)

    # 2) finalize for the answer phase
    kv.ctx_ids = ctx_ids
    kv.prefill_ids = torch.cat([sys_ids, ctx_ids], dim=1)
    if finalize:
        kv.finalize(level=cfg.level, head_ratio=cfg.head_ratio)
    if verbose:
        print(f"  [done] peak_phys_ctx={kv.peak_phys_ctx} final_cache={kv.mem_gb():.2f}GB "
              f"compactions={kv.num_compactions}", flush=True)
    return kv


@torch.inference_mode()
def answer(model, kv, query_text: str, max_new: int = 64, update_cache: bool = True) -> str:
    """Generate an answer to `query_text` against the (compressed) cache."""
    model.gen_kwargs["max_new_tokens"] = max_new
    q_ids = model.apply_template(query_text)
    return model.generate(q_ids, kv=kv, update_cache=update_cache)


@torch.inference_mode()
def evict_old_turns(kv: StreamingCache, base_phys: int, turn_budget: int):
    """Multi-turn memory bound (per_token / dense caches only): keep sink+context
    (the first `base_phys` physical positions) plus the most recent `turn_budget`
    accumulated turn tokens; drop the middle (oldest turns). StreamingLLM-style."""
    if getattr(kv, "info", {}).get("flatten"):
        # varlen (per_head finalized) cache: dense slicing is invalid. Multi-turn
        # per_head is future work; callers force per_token for SCBench.
        return
    P_total = kv.phys_total()
    turn_len = P_total - base_phys
    if turn_len <= turn_budget:
        return
    for l in range(kv.n_layers):
        kc, vc = kv.key_cache[l], kv.value_cache[l]
        kv.key_cache[l] = torch.cat([kc[:, :, :base_phys], kc[:, :, -turn_budget:]], dim=2).contiguous()
        kv.value_cache[l] = torch.cat([vc[:, :, :base_phys], vc[:, :, -turn_budget:]], dim=2).contiguous()
    # _seen_tokens stays logical so RoPE for future turns remains correct.


@torch.inference_mode()
def multiturn(model, kv, queries: List[str], max_new: int = 96,
              turn_budget: int = 2048, verbose: bool = True):
    """Run a multi-turn dialogue with cache accumulation + bounded memory.

    Each turn appends (query + answer) to the cache (update_cache=True). When the
    accumulated turn tokens exceed `turn_budget`, the oldest turns are evicted so
    total memory stays bounded -- the property that lets us survive more turns
    than a no-compaction baseline before OOM. (Dense / per_token caches.)"""
    is_stream = isinstance(kv, StreamingCache)
    base_phys = kv.phys_total() if is_stream else None  # sink + compacted context
    answers, mem_trace = [], []
    for i, q in enumerate(queries):
        a = answer(model, kv, q, max_new=max_new, update_cache=True)
        answers.append(a)
        # Only our streaming cache bounds the accumulating turns; baselines (full /
        # kvzip) accumulate without bound -> they OOM sooner. That is the point.
        if is_stream and turn_budget:
            evict_old_turns(kv, base_phys, turn_budget)
        mem_trace.append(kv._mem() if hasattr(kv, "_mem") else 0.0)
        if verbose:
            phys = kv.phys_total() if is_stream else int(kv.get_seq_length())
            print(f"  turn {i}: cache={kv._mem():.2f}GB phys={phys} | {a[:80]!r}", flush=True)
    return answers, mem_trace
