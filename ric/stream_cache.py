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
        intermediate_budget: int,
        working_max: int,
        final_budget: int,
        recent_window: int = 256,
        token_agg: str = "mean",
        light_drop_ratio: float = 0.05,
        importance: str = "score",  # "score" (reconstruction) or "recent" (StreamingLLM baseline)
    ):
        # evict_range = (sink, sink): sink = system-prompt length, context starts empty.
        super().__init__(model, evict_range=(sink_len, sink_len))
        self.intermediate_budget = int(intermediate_budget)  # B_i: tokens KEPT per committed segment
        self.working_max = int(working_max)                  # heavy-prune trigger (working window size)
        self.final_budget = int(final_budget)                # B_f: cap on total committed context
        self.recent_window = int(recent_window)
        self.token_agg = token_agg
        self.light_drop_ratio = float(light_drop_ratio)
        self.importance = importance

        # logical/physical bookkeeping for the CONTEXT region (excludes sink).
        # Physical layout: [sink | committed (frozen) | working]. `committed_len` counts
        # the frozen committed context positions; the working window is [committed_len, ctx_len).
        self.ctx_len = 0                     # physical context positions currently stored (P)
        self.committed_len = 0               # frozen committed positions (<= ctx_len)
        self.logical_ctx_len = 0             # total context tokens ever processed (>= ctx_len)
        # per-position metadata (length P), kept on CPU-cheap long tensors
        self.ctx_pos = torch.zeros((0,), dtype=torch.long, device=self.device)        # original logical pos
        self.ctx_token_ids = torch.zeros((1, 0), dtype=torch.long, device=self.device)  # for re-scoring

        # restore points for the (non-cache-updating) scoring forward
        self._restore_phys: Optional[int] = None
        self._restore_seen: Optional[int] = None

        self.num_segments = 0                # committed segments produced (heavy prunes)
        self.num_final_reprunes = 0          # global committed re-prunes at the final budget
        self.peak_phys_ctx = 0

        # StreamingLLM-style boundary protection (used by the predict-next-chunk method):
        # the first `protected_prefix_len` context positions (sink) and the last
        # `protected_suffix_len` context positions (recent / final chunk) are kept with
        # infinite importance, so only the MIDDLE is compressed by predict-target scores.
        # Best-effort: still subject to the topk budget when the budget is smaller.
        self.protected_prefix_len = 0
        self.protected_suffix_len = 0

    @property
    def num_compactions(self) -> int:
        """Back-compat alias (total prune operations) for the experiment/aggregate scripts."""
        return self.num_segments + self.num_final_reprunes

    def working_len(self) -> int:
        return self.ctx_len - self.committed_len

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

    def light_drop(self, chunk_len: int):
        """Hybrid per-chunk LIGHT drop: within the just-prefilled chunk, physically
        drop tokens whose importance is below `light_drop_ratio` * (that chunk's max).
        Removes clear junk early so the working window grows slower. Never touches the
        committed region; score-based only (callers skip this for the recency baseline)."""
        if self.light_drop_ratio <= 0 or chunk_len <= 0:
            return
        P = self.ctx_len
        start = max(P - chunk_len, self.committed_len)  # never drop committed
        imp = self._position_importance()               # [P]
        chunk_imp = imp[start:]
        if chunk_imp.numel() == 0:
            return
        thr = self.light_drop_ratio * float(chunk_imp.max())
        keep_local = (chunk_imp >= thr).nonzero(as_tuple=False).squeeze(-1)
        if keep_local.numel() == chunk_imp.numel():
            return                                      # nothing below threshold
        head = torch.arange(start, device=self.device)
        self._reindex_context(torch.cat([head, start + keep_local]))

    def prune_working_to_segment(self):
        """Heavy prune of the WORKING window down to `intermediate_budget` (top-K by
        importance, plus a protected recent tail), then freeze the survivors into the
        committed region and reset the working window. This is the per-segment
        compression triggered when the working window reaches `working_max`."""
        work_start = self.committed_len
        work_len = self.ctx_len - work_start
        if work_len <= 0:
            return
        imp = self._position_importance().clone()       # [P]
        work_imp = imp[work_start:]
        # protect the sink prefix if any of it still lives in the working window
        pp = self.protected_prefix_len - work_start
        if pp > 0:
            work_imp[:min(pp, work_len)] = float("inf")
        # protect the recent tail (and the final/recent chunk once it is known)
        rw = min(max(self.recent_window, self.protected_suffix_len), work_len)
        if rw > 0:
            work_imp[work_len - rw:] = float("inf")      # never drop the freshest working tokens
        k = min(max(self.intermediate_budget, rw), work_len)
        kept_local = torch.topk(work_imp, k).indices
        kept_local, _ = torch.sort(kept_local)           # preserve causal (logical) order
        head = torch.arange(work_start, device=self.device)
        self._reindex_context(torch.cat([head, work_start + kept_local]))
        self.committed_len = self.ctx_len                # survivors are now frozen committed
        self.num_segments += 1

    def commit_working(self):
        """Freeze the current working window into committed WITHOUT heavy pruning (used
        for the trailing partial segment at end of prefill). The final_budget cap is
        enforced separately by reprune_committed_to_final()."""
        self.committed_len = self.ctx_len

    def reprune_committed_to_final(self):
        """Final-budget enforcement: if committed context exceeds `final_budget`, keep
        the top-`final_budget` committed positions by importance (global over committed)
        and drop the rest. Any working window is left untouched."""
        if self.committed_len <= self.final_budget:
            return
        imp = self._position_importance().clone()        # [P]
        comm_imp = imp[:self.committed_len]
        # protect sink prefix and recent suffix (best-effort within the budget)
        if self.protected_prefix_len > 0:
            comm_imp[:min(self.protected_prefix_len, self.committed_len)] = float("inf")
        if self.protected_suffix_len > 0:
            comm_imp[max(0, self.committed_len - self.protected_suffix_len):] = float("inf")
        kept_comm = torch.topk(comm_imp, self.final_budget).indices
        kept_comm, _ = torch.sort(kept_comm)
        tail = torch.arange(self.committed_len, self.ctx_len, device=self.device)  # working
        self._reindex_context(torch.cat([kept_comm, tail]))
        self.committed_len = self.final_budget
        self.num_final_reprunes += 1

    def reprune_turns_to_cap(self, context_len: int, turn_cap: int):
        """Multi-turn hard compression: keep the protected initial context [0, context_len)
        intact and re-prune the committed TURN region [context_len, committed_len) down to the
        top-`turn_cap` positions by importance (scores should be repeat-based here). Any working
        tail is preserved. Used when the accumulated turns exceed the growing m*B_f cap."""
        turn_committed = self.committed_len - context_len
        if turn_committed <= turn_cap:
            return
        imp = self._position_importance().clone()              # [P]
        turn_imp = imp[context_len:self.committed_len]
        kept = torch.topk(turn_imp, turn_cap).indices
        kept, _ = torch.sort(kept)                              # preserve causal order
        head = torch.arange(context_len, device=self.device)   # protected initial context
        tail = torch.arange(self.committed_len, self.ctx_len, device=self.device)  # working (if any)
        self._reindex_context(torch.cat([head, context_len + kept, tail]))
        self.committed_len = context_len + turn_cap
        self.num_final_reprunes += 1

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
