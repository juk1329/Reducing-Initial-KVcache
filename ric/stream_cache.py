# ------------------------------------------------------------------------------
# StreamingCache: a KV cache that compacts itself *during* prefill so that the
# physically-stored KV never exceeds a budget. This is the mechanism that bounds
# PEAK prefill memory (the property KVzip / SnapKV / PyramidKV lack, because they
# materialize the full-context KV before compressing).
#
# Design (see memory note `ric-implementation-plan`):
#   * We subclass KVzip's EvictCache (DynamicCache + KVScore) to inherit dense
#     `update`, reconstruction scoring (`_get_score`), and `prune`.
#   * Storage stays DENSE during streaming. Eviction during prefill is
#     TOKEN-LEVEL (whole positions, shared across heads) -> the cache stays a
#     dense [1, n_kv, sink+P, dim] tensor, so re-scoring keeps working and real
#     memory is freed. This is the `--per_token` mode and also the shared
#     prefill behaviour for `--per_head`.
#   * `--per_head`: after streaming prefill, we additionally run KVzip's
#     per-head non-uniform eviction (`prune`) on the retained dense cache. This
#     lowers post-prefill memory further and applies per-head selection for the
#     answering phase, exactly like KVzip — but starting from an already
#     peak-bounded cache.
#
# Critical invariant: `_seen_tokens` always tracks the LOGICAL sequence length
# (sink + every context token ever processed), NOT the physical cache length.
# RoPE positions for future chunks/queries are derived from `_seen_tokens`, while
# surviving keys keep their original (now gappy) RoPE phase. This is the same
# approximation KVzip relies on for post-prune inference; here we apply it
# mid-prefill. Because we only ever DROP positions (never reorder), the physical
# array order equals increasing logical order, so flash causal attention stays
# correct.
# ------------------------------------------------------------------------------
from typing import Optional

import torch

import ric  # noqa: F401  (injects KVzip into sys.path)
from attention.kvcache import EvictCache


class StreamingCache(EvictCache):
    def __init__(
        self,
        model,
        sink_len: int,
        budget_max: int,
        budget_target: int,
        recent_window: int = 256,
        token_agg: str = "mean",
        importance: str = "score",  # "score" (reconstruction) or "recent" (StreamingLLM baseline)
    ):
        # evict_range = (sink, sink): sink = system-prompt length, context starts empty.
        super().__init__(model, evict_range=(sink_len, sink_len))
        self.budget_max = int(budget_max)
        self.budget_target = int(budget_target)
        self.recent_window = int(recent_window)
        self.token_agg = token_agg
        self.importance = importance

        # logical/physical bookkeeping for the CONTEXT region (excludes sink)
        self.ctx_len = 0                     # physical context positions currently stored (P)
        self.logical_ctx_len = 0             # total context tokens ever processed (>= ctx_len)
        # per-position metadata (length P), kept on CPU-cheap long tensors
        self.ctx_pos = torch.zeros((0,), dtype=torch.long, device=self.device)        # original logical pos
        self.ctx_token_ids = torch.zeros((1, 0), dtype=torch.long, device=self.device)  # for --rescore

        # restore points for the (non-cache-updating) scoring forward
        self._restore_phys: Optional[int] = None
        self._restore_seen: Optional[int] = None

        self.num_compactions = 0
        self.peak_phys_ctx = 0

    # --------------------------------------------------------------------- utils
    def phys_total(self) -> int:
        """Physical cache length incl. sink (= key_cache[0].shape[-2])."""
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].shape[-2]

    def mem_gb(self) -> float:
        return self._mem()

    # --------------------------------------------- scoring forward (no cache grow)
    def mark(self):
        """Record the physical length and logical _seen_tokens before a scoring
        forward, so we can undo the temporarily-appended reconstruction KV."""
        self._restore_phys = self.phys_total()
        self._restore_seen = int(self._seen_tokens)

    def restore(self):
        """Undo a scoring forward: drop the temporarily-appended reconstruction KV
        (physical truncation) and reset the logical counter.

        NOTE: we deliberately do NOT override EvictCache.slice. KVzip's slice
        truncates by LOGICAL index, which is wrong once physical != logical (after
        a compaction). We avoid that path entirely: streaming scoring forwards are
        driven manually (see stream_prefill) and undone with this method, and the
        answer phase uses update_cache=True (KVzip never slices in that case)."""
        target = self._restore_phys
        for i in range(self.n_layers):
            self.key_cache[i] = self.key_cache[i][:, :, :target].contiguous()
            self.value_cache[i] = self.value_cache[i][:, :, :target].contiguous()
        self._seen_tokens = self._restore_seen

    # ---------------------------------------------------- post-prefill bookkeeping
    def note_chunk_prefilled(self, chunk_ids: torch.Tensor):
        """Call right after a context chunk has been prefilled (update_cache=True)
        but BEFORE scoring it. Updates physical/logical context counters and the
        per-position metadata."""
        c = chunk_ids.shape[-1]
        start_logical = self.logical_ctx_len
        new_pos = torch.arange(
            start_logical, start_logical + c, dtype=torch.long, device=self.device
        )
        self.ctx_pos = torch.cat([self.ctx_pos, new_pos])
        self.ctx_token_ids = torch.cat([self.ctx_token_ids, chunk_ids.to(self.device)], dim=1)
        self.ctx_len += c
        self.logical_ctx_len += c
        self.peak_phys_ctx = max(self.peak_phys_ctx, self.ctx_len)

    def score_region_for_fresh_chunk(self, c: int):
        """Return (start_idx, end_idx) physical range of the just-prefilled chunk
        so `_get_score` scores exactly that chunk's KV."""
        end = self.sink + self.ctx_len           # current physical context end
        start = end - c
        return start, end

    # ------------------------------------------------------------------ compaction
    def need_compact(self) -> bool:
        return self.ctx_len > self.budget_max

    def _position_importance(self) -> torch.Tensor:
        """Collapse per-(layer, head, position) scores into one importance value
        per physical context position. Shape -> [P]."""
        if self.importance == "recent":
            # StreamingLLM-style baseline: importance = recency (position index).
            return self.ctx_pos.float()
        # reconstruction-based importance from KVzip scoring (self.score is a list
        # of [1, n_kv, P] tensors, one per layer).
        s = torch.stack(self.score, dim=0)  # [L, 1, n_kv, P]
        if self.token_agg == "max":
            imp = s.amax(dim=(0, 1, 2))
        else:
            imp = s.mean(dim=(0, 1, 2))
        return imp

    def compact(self):
        """Token-level compaction: keep the `budget_target` most important context
        positions (plus a protected recent window), drop the rest physically.
        Keeps the cache dense and re-scorable."""
        P = self.ctx_len
        if P <= self.budget_target:
            return
        imp = self._position_importance().clone()  # [P]

        rw = min(self.recent_window, P)
        if rw > 0:
            imp[P - rw:] = float("inf")  # never drop the freshest tokens

        k = min(max(self.budget_target, rw), P)
        kept = torch.topk(imp, k).indices
        kept, _ = torch.sort(kept)  # preserve causal (logical) order

        self._reindex_context(kept)
        self.num_compactions += 1

    def _reindex_context(self, kept_idx: torch.Tensor):
        """Physically keep only `kept_idx` (indices into [0, P)) of the context
        region across all layers, plus all score/metadata buffers."""
        P = self.ctx_len
        s, e = self.sink, self.sink + P
        for l in range(self.n_layers):
            kc, vc = self.key_cache[l], self.value_cache[l]
            sysk, ctxk = kc[:, :, :s], kc[:, :, s:e]
            sysv, ctxv = vc[:, :, :s], vc[:, :, s:e]
            self.key_cache[l] = torch.cat([sysk, ctxk[:, :, kept_idx]], dim=2).contiguous()
            self.value_cache[l] = torch.cat([sysv, ctxv[:, :, kept_idx]], dim=2).contiguous()
            if len(self.score) > l and self.score[l].shape[-1] == P:
                self.score[l] = self.score[l][:, :, kept_idx].contiguous()
        self.ctx_pos = self.ctx_pos[kept_idx]
        self.ctx_token_ids = self.ctx_token_ids[:, kept_idx]
        self.ctx_len = int(kept_idx.numel())
        # _seen_tokens (logical) intentionally unchanged.

    # ------------------------------------------------- finalize for the answer phase
    def finalize(self, level: str = "per_token", head_ratio: float = 0.5):
        """Prepare the cache for query answering / generation.

        per_token: keep dense compacted cache as-is (uniform across heads).
        per_head : run KVzip's per-head non-uniform eviction on the retained
                   dense cache -> varlen layout (lower memory + per-head selection).
        """
        if level == "per_head":
            # self.ctx_len / self.score must be aligned (they are after streaming).
            assert self.score[0].shape[-1] == self.ctx_len, (
                "score/ctx_len mismatch before per-head finalize"
            )
            # KVzip prune: ratio = retained fraction of the (already compacted) ctx.
            self.prune(ratio=head_ratio, level="pair")
        # else per_token: nothing to do; pruned stays False, attention is dense.
        return self
