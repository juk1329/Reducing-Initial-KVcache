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
from dataclasses import dataclass, field, replace
from typing import List, Optional

import torch

import ric  # noqa: F401
from model.wrapper import chunk_fn

from .stream_cache import StreamingCache


@dataclass
class RICConfig:
    level: str = "per_token"           # per_token | per_head
    intermediate_budget: int = 2048    # B_i: tokens KEPT per committed segment (heavy-prune target)
    working_max: int = 4096            # heavy-prune trigger: working window size that fires a segment prune
    final_budget: int = 8192           # B_f: cap on total committed context (multiple of B_i)
    recent_window: int = 256           # most-recent tokens never evicted during a prune
    chunk: int = 2048                  # prefill + scoring chunk size
    token_agg: str = "mean"            # mean | max  (collapse head/layer scores -> per position)
    light_drop_ratio: float = 0.05     # Hybrid per-chunk light drop: drop score < ratio * chunk_max (0 disables)
    importance: str = "score"          # score (reconstruction) | recent (StreamingLLM baseline)
    rescore_working: bool = True       # re-score the working window with the repeat prompt before each heavy prune
    head_ratio: float = 0.5            # per_head finalize: retained fraction after streaming
    prev_tail: int = 8                 # hint tokens carried from the previous chunk
    use_predict_prompt: bool = False   # intermediate (chunk/segment) prunes use a PREDICT prompt
                                       # instead of the repeat prompt; the final compaction always
                                       # uses the repeat prompt (reconstruction, like kvzip)
    predict_prompt_version: int = 1    # 1: "Predict the entire context:"  2: "Predict the upcoming context:"
    predict_target: str = "self"       # self: score a chunk with a prompt on ITSELF (v1/v2);
                                       # next_chunk: score a chunk by how much the REAL next chunk
                                       # attends to it (true causal-prediction signal). next_chunk
                                       # also keeps the first chunk (sink) and last chunk (recent)
                                       # uncompressed and only predict-scores the middle chunks.
    combine_repeat: bool = False       # intermediate scoring = COMBINE(repeat, predict) per position
                                       # (recovers repeat's retrieval signal while keeping predict's).
    combine_mode: str = "max"          # max | wsum  (how to merge the two score tensors)
    combine_alpha: float = 0.5         # wsum weight on the PREDICT signal: alpha*predict+(1-alpha)*repeat
    sink_tokens: int = 4               # predict-next boundary protection: # of leading context tokens
                                       # kept as an attention SINK (StreamingLLM-style, small). The
                                       # recent tail uses `recent_window`. Small caps so the boundary
                                       # protection does not eat the prune budget at aggressive ratios.

    def __post_init__(self):
        if self.working_max <= self.intermediate_budget:
            # the trigger must exceed the kept size, else the heavy prune never fires
            self.working_max = self.intermediate_budget + max(1, self.intermediate_budget)
        # NOTE: final_budget need NOT be a multiple of intermediate_budget -- committed
        # accumulates in B_i steps and reprune_committed_to_final() caps it at exactly
        # final_budget via top-k, so any integer final_budget is valid.


def derive_ric_budgets(comp_ratio: float, ctx_len: int, chunk: int,
                       recent_window: int = 256):
    """Map a target OVERALL retained-fraction `comp_ratio` (r) and context length L to
    RIC streaming budgets, so ours' overall retained fraction ~= r (matching kvzip's
    `--ratio r`). Returns (final_budget, intermediate_budget, working_max).

      final_budget = round(r * L)             # committed cap = retained tokens
      working_max  = 2 * chunk (fixed)        # bounds streaming peak independent of L
      intermediate_budget = round(r * working_max)
                                              # per-segment keep fraction = r, so segments
                                              # accumulate to ~r*L (final cap rarely binds)

    Edge cases: r >= 1 -> no compression (keep all, behaves like full prefill);
                r <= 0 -> keep ~nothing (a recent_window floor so the run is still valid).
    Peak then scales ~ r*L + working_max, i.e. ~r x lower than kvzip's full-L peak."""
    r = float(comp_ratio)
    L = int(ctx_len)
    if r >= 1.0:                                  # no compression: behave like full prefill
        return L, L, L + chunk
    W = 2 * chunk
    fb = int(round(r * L))
    if fb <= 0:                                   # r ~ 0: keep ~nothing (floor to recent_window)
        fb = max(1, int(recent_window))
        return fb, fb, max(W, fb + chunk)
    Bi = max(1, min(int(round(r * W)), fb))       # per-segment kept size; never exceed fb
    if Bi >= W:                                   # keep the trigger above the kept size
        W = Bi + chunk
    return fb, Bi, W


def derive_ric_budgets_for_ratio(comp_ratio: float, ctx_len: int, chunk: int,
                                 recent_window: int = 256,
                                 head_ratio_target: float = 0.5):
    """PER-HEAD ratio split (used by the NIAH grid, level=per_head).

    ours bounds prefill peak by dropping whole POSITIONS during streaming, then the
    final per-head prune drops per-head entries among the survivors. To hit an OVERALL
    retained fraction r (matching kvzip's `--ratio r`) while still doing a genuine
    per-head varlen selection, keep a LOOSER positional budget and let the per-head
    prune finish the job:

        f               = min(1, r / head_ratio_target)   # positional fraction kept (peak ~ f*L)
        head_ratio_eff  = r / f                            # per-head retained fraction
        overall         = f * head_ratio_eff = r

    With head_ratio_target=0.5: r<=0.5 -> f=2r, head_ratio_eff=0.5 (peak ~ 2r*L, ~2x
    below kvzip's full-L); r>=0.5 -> f=1 (keep all positions), head_ratio_eff=r (peak ~ L,
    same as kvzip -- no positional headroom left).

    Returns (final_budget, intermediate_budget, working_max, head_ratio_eff, recent_window_eff).
    Edge cases: r>=1 -> no compression; r<=0 -> keep ~nothing (recent_window floor, hr~0)."""
    r = float(comp_ratio)
    L = int(ctx_len)
    if r >= 1.0:
        return L, L, L + chunk, 1.0, min(recent_window, L)
    if r <= 0.0:
        fb = max(1, min(int(recent_window), L))
        return fb, fb, max(2 * chunk, fb + chunk), 0.0, min(recent_window, fb)
    f = min(1.0, r / float(head_ratio_target))
    head_ratio_eff = r / f                                  # 0.5 for r<=0.5, else r
    fb = max(1, int(round(f * L)))
    W = 2 * chunk
    Bi = max(1, min(int(round(f * W)), fb))
    if Bi >= W:
        W = Bi + chunk
    rw = min(int(recent_window), max(1, fb // 2))           # never protect more than half the budget
    return fb, Bi, W, head_ratio_eff, rw


PREDICT_PROMPTS = {
    1: "\n\nPredict the entire context:",
    2: "\n\nPredict the upcoming context:",
}


def build_predict(model, a_ids: torch.Tensor, version: int = 1):
    """Predict-prompt variant for intermediate (peak-reducing) scoring. Same target
    assembly as build_repeat (`[q] + postfix + chunk`), but the instruction asks the
    model to PREDICT rather than reconstruct; no `starting with ...` continuation."""
    q = model.encode(PREDICT_PROMPTS.get(int(version), PREDICT_PROMPTS[1]))
    return torch.cat([q, model.postfix_ids, a_ids], dim=1)


def _intermediate_builder(model, cfg: "RICConfig"):
    """Prompt builder used for chunk/segment (intermediate) scoring: predict if
    enabled, else repeat. Signature: builder(a_ids, prev_tail) -> input_ids."""
    if cfg.use_predict_prompt:
        ver = cfg.predict_prompt_version
        return lambda a_ids, prev_tail: build_predict(model, a_ids, ver)
    return lambda a_ids, prev_tail: build_repeat(model, a_ids, prev_tail)


def _repeat_builder(model, cfg: "RICConfig"):
    """Prompt builder for the FINAL reconstruction-scored compaction (always repeat)."""
    return lambda a_ids, prev_tail: build_repeat(model, a_ids, prev_tail)


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
def _score_region_with_query(model, kv: StreamingCache, query_ids: torch.Tensor,
                             start: int, end: int):
    """Score the physical KV region [start, end) by how much `query_ids` attends to it.
    `query_ids` are appended to the cache for the forward (their own keys become the
    `-q_len:` tail KVzip's _get_score expects) and undone via mark/restore. The query
    may be a repeat target, a predict prompt, or the real next chunk -- KVzip handles
    a query length that differs from the scored-region length."""
    base = model.model.model  # HF base model; avoids ModelKVzip.__call__'s slice
    kv.start_idx, kv.end_idx = start, end
    kv.mark()
    kv.get_score = True
    base(query_ids.to(model.device), past_key_values=kv)
    kv.get_score = False
    kv.restore()


def _combine_scores(rep, pred, mode: str, alpha: float):
    """Element-wise combine of two per-(layer,head,pos) score slices (both softmax
    attention weights in [0,1], so directly comparable)."""
    if mode == "wsum":
        return alpha * pred + (1.0 - alpha) * rep
    return torch.maximum(rep, pred)  # default: keep a position if EITHER signal flags it


@torch.inference_mode()
def _score_region_combined(model, kv: StreamingCache, repeat_query, predict_query,
                           start: int, end: int, mode: str, alpha: float):
    """Score region [start,end) by COMBINING a repeat forward and a predict forward.
    Runs repeat (appends region scores), captures + truncates them, runs predict
    (appends region scores), then overwrites the appended slice with the combination
    -- so the single kv.score buffer ends up holding the combined importance."""
    base_len = [kv.score[l].shape[-1] for l in range(kv.n_layers)]
    _score_region_with_query(model, kv, repeat_query, start, end)
    rep = [kv.score[l][:, :, base_len[l]:].clone() for l in range(kv.n_layers)]
    for l in range(kv.n_layers):
        kv.score[l] = kv.score[l][:, :, :base_len[l]].contiguous()
    _score_region_with_query(model, kv, predict_query, start, end)
    for l in range(kv.n_layers):
        comb = _combine_scores(rep[l], kv.score[l][:, :, base_len[l]:], mode, alpha)
        kv.score[l] = torch.cat([kv.score[l][:, :, :base_len[l]], comb], dim=-1).contiguous()


def _predict_query_fresh(model, cfg: "RICConfig", chunk, prev_tail,
                         next_chunk: Optional[torch.Tensor]):
    """The PREDICT query for a fresh chunk: the real next chunk (next_chunk target,
    repeat fallback for the last chunk) or the predict prompt on this chunk (self)."""
    if cfg.predict_target == "next_chunk":
        return next_chunk if next_chunk is not None else build_repeat(model, chunk, prev_tail)
    return build_predict(model, chunk, cfg.predict_prompt_version)


@torch.inference_mode()
def _score_fresh_chunk(model, kv: StreamingCache, chunk: torch.Tensor,
                       prev_tail: Optional[torch.Tensor], cfg: "RICConfig",
                       next_chunk: Optional[torch.Tensor] = None):
    """Append per-position importance scores for the just-prefilled chunk.

    combine_repeat:           query importance = COMBINE(repeat-self, predict).
    predict_target=='next_chunk': query = the REAL next chunk (repeat fallback for the last chunk).
    otherwise:                query = the intermediate prompt builder (predict-self or repeat)."""
    c = chunk.shape[-1]
    start, end = kv.score_region_for_fresh_chunk(c)
    if cfg.use_predict_prompt and cfg.combine_repeat:
        rep_q = build_repeat(model, chunk, prev_tail)
        pred_q = _predict_query_fresh(model, cfg, chunk, prev_tail, next_chunk)
        _score_region_combined(model, kv, rep_q, pred_q, start, end, cfg.combine_mode, cfg.combine_alpha)
    elif cfg.use_predict_prompt and cfg.predict_target == "next_chunk":
        query = next_chunk if next_chunk is not None else build_repeat(model, chunk, prev_tail)
        _score_region_with_query(model, kv, query, start, end)
    else:
        _score_region_with_query(model, kv, _intermediate_builder(model, cfg)(chunk, prev_tail), start, end)


@torch.inference_mode()
def _rescore_region(model, kv: StreamingCache, cfg: RICConfig, lo: int, builder):
    """Re-score context positions [lo, ctx_len) with `builder` (repeat or predict),
    preserving scores for [0, lo). The region is re-run chunk-by-chunk over the
    (possibly gappy, light-dropped) surviving tokens; survivors keep their original
    (gappy) RoPE phase -- the same approximation KVzip's post-prune inference relies on.

    `builder(a_ids, prev_tail) -> input_ids`. Used for both the per-segment working
    rescore (lo=committed_len) and the final full-context repeat rescore (lo=0)."""
    base = model.model.model
    region_ids = kv.ctx_token_ids[:, lo:]
    if region_ids.shape[-1] == 0:
        return
    # keep scores below `lo`; drop the region slice so it is recomputed below.
    for l in range(kv.n_layers):
        kv.score[l] = kv.score[l][:, :, :lo].contiguous()
    kv.get_score = False
    prev_tail = None
    pos = lo
    for w in chunk_fn(region_ids, cfg.chunk):
        c = w.shape[-1]
        kv.start_idx = kv.sink + pos
        kv.end_idx = kv.sink + pos + c
        score_ids = builder(w, prev_tail)
        kv.mark()
        kv.get_score = True
        base(score_ids, past_key_values=kv)
        kv.get_score = False
        kv.restore()
        prev_tail = w[:, -cfg.prev_tail:]
        pos += c
    assert kv.score[0].shape[-1] == kv.ctx_len


@torch.inference_mode()
def rescore_working(model, kv: StreamingCache, cfg: RICConfig,
                    next_chunk: Optional[torch.Tensor] = None):
    """Re-score the WORKING window only ([committed_len, ctx_len)) before the heavy
    per-segment prune. Committed scores are preserved.

    combine_repeat: re-score working with BOTH repeat and predict, then combine per position.
    predict_target=='next_chunk' (with a next chunk available): score the whole working
      region in ONE forward by the real next chunk's attention.
    otherwise: re-score chunk-by-chunk with the intermediate prompt (predict-self/repeat)."""
    lo = kv.committed_len
    if cfg.use_predict_prompt and cfg.combine_repeat:
        if kv.ctx_len - lo <= 0:
            return
        # repeat scores for the working region
        _rescore_region(model, kv, cfg, lo, _repeat_builder(model, cfg))
        rep = [kv.score[l][:, :, lo:].clone() for l in range(kv.n_layers)]
        for l in range(kv.n_layers):
            kv.score[l] = kv.score[l][:, :, :lo].contiguous()
        # predict scores for the working region
        if cfg.predict_target == "next_chunk" and next_chunk is not None:
            _score_region_with_query(model, kv, next_chunk, kv.sink + lo, kv.sink + kv.ctx_len)
        else:
            _rescore_region(model, kv, cfg, lo, _intermediate_builder(model, cfg))
        for l in range(kv.n_layers):
            comb = _combine_scores(rep[l], kv.score[l][:, :, lo:], cfg.combine_mode, cfg.combine_alpha)
            kv.score[l] = torch.cat([kv.score[l][:, :, :lo], comb], dim=-1).contiguous()
        assert kv.score[0].shape[-1] == kv.ctx_len
        return
    if cfg.use_predict_prompt and cfg.predict_target == "next_chunk" and next_chunk is not None:
        if kv.ctx_len - kv.committed_len <= 0:
            return
        for l in range(kv.n_layers):
            kv.score[l] = kv.score[l][:, :, :kv.committed_len].contiguous()
        _score_region_with_query(model, kv, next_chunk,
                                 kv.sink + kv.committed_len, kv.sink + kv.ctx_len)
        assert kv.score[0].shape[-1] == kv.ctx_len
    else:
        _rescore_region(model, kv, cfg, kv.committed_len, _intermediate_builder(model, cfg))


@torch.inference_mode()
def rescore_full_repeat(model, kv: StreamingCache, cfg: RICConfig):
    """Re-score the ENTIRE retained context [0, ctx_len) with the REPEAT prompt, just
    before the final (per-head) compaction, so the cache actually used for queries is
    reconstruction-scored like kvzip. Only needed when intermediate scoring used the
    predict prompt (otherwise scores are already repeat-based)."""
    _rescore_region(model, kv, cfg, 0, _repeat_builder(model, cfg))


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
        intermediate_budget=cfg.intermediate_budget, working_max=cfg.working_max,
        final_budget=cfg.final_budget, recent_window=cfg.recent_window,
        token_agg=cfg.token_agg, light_drop_ratio=cfg.light_drop_ratio,
        importance=cfg.importance,
    )

    # 0) prefill the system prompt (the protected sink region)
    model(sys_ids, kv, update_cache=True)
    kv.init_score()
    kv.get_score = False
    kv.pruned = False

    # 1) stream the context
    # predict-next-chunk mode keeps the first chunk (sink) and last chunk (recent)
    # uncompressed and only predict-scores the middle chunks.
    predict_next = (cfg.importance == "score" and cfg.use_predict_prompt
                    and cfg.predict_target == "next_chunk")
    chunks = list(chunk_fn(ctx_ids, cfg.chunk))
    n = len(chunks)
    prev_tail = None
    for ci, chunk in enumerate(chunks):
        next_chunk = chunks[ci + 1] if ci + 1 < n else None
        model(chunk, kv, update_cache=True)             # prefill (dense append -> working window)
        kv.note_chunk_prefilled(chunk)
        if predict_next and ci == 0:
            # small StreamingLLM-style sink (NOT the whole first chunk) so protection
            # does not consume the prune budget; the rest of the first chunk competes.
            kv.protected_prefix_len = min(cfg.sink_tokens, chunk.shape[-1])
        if predict_next and ci == n - 1:
            # small recent window protected; the last chunk (no next) is scored by repeat
            # via the combine fallback, so it need not be fully kept.
            kv.protected_suffix_len = min(cfg.recent_window, chunk.shape[-1])
        if cfg.importance == "score":
            _score_fresh_chunk(model, kv, chunk, prev_tail, cfg, next_chunk)
            # light drop on MIDDLE chunks only when predict_next (keep sink/recent intact)
            if cfg.light_drop_ratio > 0 and not (predict_next and (ci == 0 or ci == n - 1)):
                kv.light_drop(chunk.shape[-1])                # Hybrid: drop very-low-score tokens
        prev_tail = chunk[:, -cfg.prev_tail:]
        if kv.working_len() >= cfg.working_max:               # heavy prune of the working window
            if cfg.importance == "score" and cfg.rescore_working:
                rescore_working(model, kv, cfg, next_chunk)
            kv.prune_working_to_segment()                     # top-B_i -> commit & freeze, reset window
            kv.reprune_committed_to_final()                   # enforce final_budget over committed
        if verbose:
            print(f"  chunk {ci}: logical_ctx={kv.logical_ctx_len} phys_ctx={kv.ctx_len} "
                  f"committed={kv.committed_len} working={kv.working_len()} "
                  f"cache={kv.mem_gb():.2f}GB segments={kv.num_segments} "
                  f"reprunes={kv.num_final_reprunes}", flush=True)

    # 1b) flush the trailing partial working window into committed (capped by final_budget)
    if kv.working_len() > 0:
        if cfg.importance == "score" and cfg.rescore_working:
            rescore_working(model, kv, cfg, None)   # no next chunk -> repeat-target fallback
        kv.commit_working()
        kv.reprune_committed_to_final()

    # 1c) if intermediate scoring used the predict prompt, re-score the full retained
    #     context with the REPEAT prompt so the final (per-head) compaction below is
    #     reconstruction-scored like kvzip. No-op when repeat was used throughout.
    if finalize and cfg.importance == "score" and cfg.use_predict_prompt:
        rescore_full_repeat(model, kv, cfg)

    # 2) finalize for the answer phase
    kv.ctx_ids = ctx_ids
    kv.prefill_ids = torch.cat([sys_ids, ctx_ids], dim=1)
    if finalize:
        kv.finalize(level=cfg.level, head_ratio=cfg.head_ratio)
    if verbose:
        print(f"  [done] peak_phys_ctx={kv.peak_phys_ctx} committed={kv.committed_len} "
              f"final_cache={kv.mem_gb():.2f}GB segments={kv.num_segments} "
              f"reprunes={kv.num_final_reprunes}", flush=True)
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


@torch.inference_mode()
def multiturn_compress(model, kv: StreamingCache, queries: List[str], golds: List[str],
                       cfg: RICConfig, score_fn, base_final_budget: int,
                       max_new: int = 64, max_turns: int = 0, consec_zero_stop: int = 5,
                       turn_ratio: float = -1.0, turn_cap_mult: int = 2, rescore_every: int = 8,
                       device: int = 0, verbose: bool = True):
    """RIC multi-turn compression: treat each (query+answer) turn as a chunk and apply the
    same streaming compression used for the initial context.

    Per turn: generate the answer (appends turn KV), register it as a chunk, score it
    (predict-self + repeat for combine; repeat-only otherwise), and prune it.

    Two pruning modes:
      * DYNAMIC (turn_ratio >= 0, default for SCBench): keep the top `round(turn_ratio*Lt)`
        of THIS turn (Lt = turn length) -> the whole session (context + every turn) is
        compressed at the same retained ratio. Turn region grows at turn_ratio x the raw
        rate (constant-factor reduction vs kvzip), with only small per-turn ops (no big
        cap-region re-score) so the multi-turn PEAK stays low.
      * FIXED CAP (turn_ratio < 0, legacy): keep each turn to 0.1*B_base, hold the committed
        turn region at a fixed `cap = turn_cap_mult*B_base` (plateau).
    The initial context is the protected committed prefix `context_len`, never touched.

    Stops at OOM (stop_reason='oom', records max tokens) or `consec_zero_stop` consecutive
    zero-acc turns (stop_reason='acc_collapse'); else 'completed'."""
    from .mem import run_oom_safe, peak_gb, reset_peak
    assert isinstance(kv, StreamingCache), "multiturn_compress needs a StreamingCache (per_token)"
    B_base = int(base_final_budget)
    B_turn = max(1, int(round(0.1 * B_base)))
    cap = max(1, int(turn_cap_mult) * B_base)        # FIXED-cap mode bound on the turn region
    dynamic = turn_ratio is not None and turn_ratio >= 0.0
    context_len = int(kv.committed_len)              # frozen initial context (== ctx_len here)
    # DYNAMIC mode: keep recency protection tiny so the per-turn ratio budget actually binds
    # (a large recent_window would otherwise preserve small turns whole).
    turn_recent = min(cfg.recent_window, 8) if dynamic else cfg.recent_window
    cfg_turn = replace(cfg, predict_target="self", intermediate_budget=B_turn,
                       recent_window=turn_recent)
    # turn pruning manages recency via recent_window; the context is protected via context_len.
    kv.protected_prefix_len = 0
    kv.protected_suffix_len = 0
    kv.recent_window = turn_recent
    n = max_turns if (max_turns and max_turns > 0) else len(queries)
    reprunes = {"n": 0}

    def _do_turn(q):
        len_before = kv.prefill_ids.shape[1]
        ans = answer(model, kv, q, max_new=max_new, update_cache=True)   # appends (query+answer) KV
        turn_ids = kv.prefill_ids[:, len_before:]                        # exactly [query+answer]
        Lt = turn_ids.shape[-1]
        if Lt > 0:
            kv.note_chunk_prefilled(turn_ids)                            # register turn as a chunk
            _score_fresh_chunk(model, kv, turn_ids, None, cfg_turn)      # predict-self+repeat / repeat
            if cfg.light_drop_ratio > 0:
                kv.light_drop(Lt)
            if dynamic:
                # compress THIS turn to round(turn_ratio*Lt) by importance, then commit.
                kv.intermediate_budget = max(1, int(round(turn_ratio * Lt)))
                kv.prune_working_to_segment()
            else:
                kv.intermediate_budget = B_turn
                kv.prune_working_to_segment()                            # keep top 0.1*B_base
                if (kv.committed_len - context_len) > cap:               # FIXED-cap bound
                    if rescore_every > 0 and reprunes["n"] % rescore_every == 0:
                        _rescore_region(model, kv, cfg_turn, context_len,
                                        _repeat_builder(model, cfg_turn))
                    kv.reprune_turns_to_cap(context_len, cap)
                    reprunes["n"] += 1
        return ans

    reset_peak(device)
    turns_rec, stop_reason, consec_zero = [], "completed", 0
    for t in range(n):
        q = queries[t % len(queries)]
        gold = golds[t % len(golds)] if golds else ""
        res, info = run_oom_safe(lambda: _do_turn(q), device=device)
        if info["status"] != "ok":
            stop_reason = info["status"]  # 'oom' or 'error'
            if verbose:
                print(f"  turn {t}: {stop_reason.upper()} (survived {t} turns, "
                      f"~{kv.sink + kv.logical_ctx_len} logical tokens)", flush=True)
            break
        acc = float(score_fn(res, gold))
        consec_zero = consec_zero + 1 if acc == 0.0 else 0
        rec = {"turn": t, "cache_gb": float(kv._mem()), "phys": int(kv.phys_total()),
               "logical_tokens": int(kv.sink + kv.logical_ctx_len),
               "committed_turns": int(kv.committed_len - context_len),
               "n_reprunes": reprunes["n"], "acc": acc,
               "answer": res[:120], "gold": str(gold)[:80]}
        turns_rec.append(rec)
        if verbose and (t % 10 == 0 or t == n - 1):
            mode = f"turn_ratio={turn_ratio}" if dynamic else f"cap{cap} reprunes={reprunes['n']}"
            print(f"  turn {t}: cache={rec['cache_gb']:.2f}GB phys={rec['phys']} "
                  f"ctx_turns={rec['committed_turns']} {mode} acc={acc:.2f} | {res[:50]!r}",
                  flush=True)
        if consec_zero >= consec_zero_stop:
            stop_reason = "acc_collapse"
            if verbose:
                print(f"  turn {t}: acc collapse ({consec_zero} consecutive 0s) -> stop", flush=True)
            break

    return {
        "turns": turns_rec,
        "stop_reason": stop_reason,
        "survived_turns": len(turns_rec),
        "peak_during_turns_gb": peak_gb(device),
        "final_cache_gb": (turns_rec[-1]["cache_gb"] if turns_rec else None),
        "max_logical_tokens": (turns_rec[-1]["logical_tokens"] if turns_rec
                               else int(kv.sink + kv.logical_ctx_len)),
        "mean_acc": (round(sum(r["acc"] for r in turns_rec) / len(turns_rec), 4)
                     if turns_rec else None),
        "B_base": B_base, "turn_ratio": (turn_ratio if dynamic else None),
        "turn_cap": (None if dynamic else cap), "n_turn_reprunes": reprunes["n"],
    }
