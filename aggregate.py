#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Aggregate recorded runs into paper-ready tables.
#
# Reads results/niah/*.json and results/scbench/*.json (written by the run
# scripts) and emits:
#   results/summary_niah.md      - peak-memory & accuracy tables + OOM headline
#   results/summary_niah.csv     - long format (one row per (method, ctx_len))
#   results/summary_scbench.md   - multi-turn survival / accuracy table
#   results/summary_scbench.csv  - long format (one row per method)
#
# Re-run any time after running experiments:  conda run -n jk python aggregate.py
# When several runs share a config, the most recent (by timestamp) is used.
# ------------------------------------------------------------------------------
import argparse
import csv
import glob
import json
import os


def load_records(d):
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(p) as f:
                r = json.load(f)
            r["_file"] = p
            recs.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {p}: {e}")
    recs.sort(key=lambda r: r.get("timestamp", ""))  # latest last -> wins in dicts
    return recs


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


# ----------------------------------------------------------------------- NIAH
def aggregate_niah(results_dir):
    recs = load_records(os.path.join(results_dir, "niah"))
    if not recs:
        return None, []
    # group by model; within a model, key method by (method, level, budget)
    models = {}
    long_rows = []
    for r in recs:
        a = r.get("args", {})
        model = a.get("model", (r.get("env", {}) or {}).get("model_name", "?"))
        cr = a.get("comp_ratio")
        mkey = f"{a.get('method')}/{a.get('level')}"
        if cr is not None and cr >= 0:
            mkey += f"(cr={cr})"            # compression-ratio sweep: r applied to both methods
        elif a.get("method") == "kvzip":
            mkey += f"(r={a.get('ratio')})"
        elif a.get("method") in ("ours", "streamllm"):
            mkey += f"(ib={a.get('intermediate_budget')},wm={a.get('working_max')},fb={a.get('final_budget')})"
        m = models.setdefault(model, {})
        m.setdefault(mkey, {})  # mkey -> {ctx_len: row}
        for row in r.get("rows", []):
            m[mkey][row["ctx_len"]] = row  # latest run wins
            long_rows.append({
                "model": model, "method": a.get("method"), "level": a.get("level"),
                "comp_ratio": row.get("comp_ratio", cr),
                # per-length effective budgets (recorded per row; vary with ctx_len under a sweep)
                "intermediate_budget": row.get("intermediate_budget", a.get("intermediate_budget")),
                "working_max": row.get("working_max", a.get("working_max")),
                "final_budget": row.get("final_budget", a.get("final_budget")),
                "ratio": row.get("kvzip_ratio", a.get("ratio")),
                "ctx_len": row["ctx_len"], "true_len": row.get("true_len"),
                "prefill_status": row.get("prefill_status"),
                "prefill_peak_gb": row.get("prefill_peak_gb"),
                "answer_peak_gb": row.get("answer_peak_gb"),
                "cache_gb": row.get("cache_gb"),
                "num_compactions": row.get("num_compactions"),
                "num_segments": row.get("num_segments"),
                "peak_phys_ctx": row.get("peak_phys_ctx"),
                "acc": row.get("acc"),
            })

    md = ["# NIAH length-ladder summary",
          "",
          "Peak GPU memory (GB) during **prefill** vs context length. `OOM` = ran out "
          "of memory; blank = not run. Lower peak / further reach is better.",
          ""]
    for model, methods in sorted(models.items()):
        ctx_lens = sorted({L for mk in methods.values() for L in mk})
        gpu = next((r.get("env", {}).get("gpu_total_gb") for r in recs), None)
        kv1k = next((r.get("env", {}).get("kv_gb_per_1k_tokens") for r in recs), None)
        md += [f"## {model}  (GPU ~{gpu}GB, KV {kv1k} GB/1k tok)", ""]

        # peak table
        headers = ["method \\ ctx_len"] + [str(L) for L in ctx_lens]
        prows = []
        for mk, byL in methods.items():
            cells = []
            for L in ctx_lens:
                row = byL.get(L)
                if row is None:
                    cells.append("")
                elif row["prefill_status"] == "ok":
                    cells.append(fmt(row["prefill_peak_gb"]))
                else:
                    cells.append("OOM")
            prows.append([mk] + cells)
        md += ["### Prefill peak memory (GB)", md_table(headers, prows), ""]

        # accuracy table
        arows = []
        for mk, byL in methods.items():
            cells = []
            for L in ctx_lens:
                row = byL.get(L)
                cells.append("" if (row is None or row.get("acc") is None) else fmt(row["acc"], 2))
            arows.append([mk] + cells)
        md += ["### Needle accuracy (0-1)", md_table(headers, arows), ""]

        # headline: max OOM-free ctx per method
        hrows = []
        for mk, byL in methods.items():
            ok = [r for r in byL.values() if r["prefill_status"] == "ok"]
            ooms = [r for r in byL.values() if r["prefill_status"] == "oom"]
            best = max(ok, key=lambda r: r["true_len"], default=None)
            hrows.append([
                mk,
                best["ctx_len"] if best else "",
                best.get("true_len") if best else "",
                fmt(best["prefill_peak_gb"]) if best else "",
                fmt(best.get("acc"), 2) if best else "",
                best.get("num_segments") if best else "",
                (min(r["ctx_len"] for r in ooms) if ooms else ""),
            ])
        md += ["### Headline: how far before OOM",
               md_table(["method", "max_ok_ctx", "max_ok_tok", "peak_gb@max",
                         "acc@max", "segments@max", "first_oom_ctx"], hrows), ""]
    return "\n".join(md), long_rows


# --------------------------------------------------------------- NIAH grid
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


_METHOD_ORDER = {"kvzip": 0, "ours": 1, "ours_predict_v1": 2, "ours_predict_v2": 3,
                 "ours_predict_next": 4, "ours_combine": 5}


def aggregate_niah_grid(results_dir):
    """Compression-ratio grid (run_niah_grid.py): per (ctx_len, ratio) compare prefill
    peak (ours vs kvzip) and needle accuracy across 11 positions, plus per-position
    heatmaps. Peak is depth-independent -> averaged over needle positions."""
    recs = load_records(os.path.join(results_dir, "niah_grid"))
    if not recs:
        return None, []
    # latest run wins per cell (recs are sorted oldest->newest by load_records)
    cells = {}
    for r in recs:
        for row in r.get("rows", []):
            model = row.get("model") or r.get("args", {}).get("model", "?")
            row = dict(row)
            row["model"] = model
            cells[(model, row["method"], row["ctx_len"], row["depth"], row["comp_ratio"])] = row
    allrows = list(cells.values())

    long_keys = ["model", "method", "base_method", "ctx_len", "true_len", "depth",
                 "needle_quantile", "comp_ratio", "level", "use_predict_prompt",
                 "predict_prompt_version", "predict_target", "head_ratio_eff", "final_budget",
                 "intermediate_budget", "working_max", "num_segments",
                 "prefill_status", "prefill_peak_gb", "cache_gb", "acc"]
    long_rows = [{k: row.get(k) for k in long_keys}
                 for row in sorted(allrows, key=lambda r: (r["model"], r["method"],
                                   r["ctx_len"], r["comp_ratio"], r["depth"]))]

    md = ["# NIAH compression-ratio grid (per-head)", "",
          "All methods run with `level=per_head`; the NIAH context for each "
          "(ctx_len, needle position) is generated once and fed identically to every "
          "method/ratio. Peak = prefill GPU memory (depth-independent, averaged over "
          "needle positions). Accuracy = mean over the needle quantiles.", ""]
    for model in sorted({r["model"] for r in allrows}):
        mr = [r for r in allrows if r["model"] == model]
        methods = sorted({r["method"] for r in mr},
                         key=lambda m: (_METHOD_ORDER.get(m, 9), m))
        ctxs = sorted({r["ctx_len"] for r in mr})
        ratios = sorted({r["comp_ratio"] for r in mr})
        quants = sorted({r["needle_quantile"] for r in mr})
        md += [f"## {model}", ""]

        for L in ctxs:
            # prefill peak (GB) + ours/kvzip ratio
            headers = ["ratio \\ method"] + methods + ["ours/kvzip"]
            prows = []
            for rr in ratios:
                pk = {m: _mean([c["prefill_peak_gb"] for c in mr
                                if c["ctx_len"] == L and c["comp_ratio"] == rr
                                and c["method"] == m and c["prefill_status"] == "ok"])
                      for m in methods}
                kp, op = pk.get("kvzip"), pk.get("ours")
                rcol = (op / kp) if (op and kp) else None
                prows.append([rr] + [fmt(pk[m], 3) for m in methods] + [fmt(rcol, 2)])
            md += [f"### ctx_len={L} — prefill peak (GB)", md_table(headers, prows), ""]

            # needle accuracy (mean over positions)
            ah = ["ratio \\ method"] + methods
            arows = []
            for rr in ratios:
                cells_acc = [_mean([c["acc"] for c in mr
                                    if c["ctx_len"] == L and c["comp_ratio"] == rr
                                    and c["method"] == m and c["acc"] is not None])
                             for m in methods]
                arows.append([rr] + [fmt(x, 2) for x in cells_acc])
            md += [f"### ctx_len={L} — needle accuracy (mean over {len(quants)} positions)",
                   md_table(ah, arows), ""]

        # per-position heatmaps (rows=ratio, cols=needle quantile)
        md += ["### Per-position accuracy heatmaps (rows=ratio, cols=needle quantile)", ""]
        for L in ctxs:
            for m in methods:
                hh = ["r \\ q"] + [str(q) for q in quants]
                hrows = []
                for rr in ratios:
                    vals = [_mean([c["acc"] for c in mr
                                   if c["ctx_len"] == L and c["comp_ratio"] == rr
                                   and c["method"] == m and c["needle_quantile"] == q
                                   and c["acc"] is not None])
                            for q in quants]
                    hrows.append([rr] + [fmt(x, 1) for x in vals])
                md += [f"#### {model} ctx={L} {m}", md_table(hh, hrows), ""]
    return "\n".join(md), long_rows


# -------------------------------------------------------------------- SCBench
def aggregate_scbench(results_dir):
    recs = load_records(os.path.join(results_dir, "scbench"))
    if not recs:
        return None, []
    groups = {}  # (model, data) -> {method_variant: (summary, cache_growth)}
    long_rows = []
    for r in recs:
        a = r.get("args", {})
        model = a.get("model", "?")
        data = a.get("data", "?")
        s = r.get("summary", {}) or {}
        mkey = s.get("method_variant") or a.get("method")
        # per-turn cache growth: cache_gb at first / mid / last turn + slope (GB/turn)
        tr = r.get("turns", []) or []
        cgs = [t.get("cache_gb") for t in tr if t.get("cache_gb") is not None]
        growth = {"c0": (cgs[0] if cgs else None),
                  "cmid": (cgs[len(cgs) // 2] if cgs else None),
                  "clast": (cgs[-1] if cgs else None),
                  "slope": (round((cgs[-1] - cgs[0]) / max(1, len(cgs) - 1), 5) if len(cgs) > 1 else None)}
        groups.setdefault((model, data), {})[mkey] = (s, growth)
        long_rows.append({
            "model": model, "data": data, "method": a.get("method"),
            "method_variant": mkey, "comp_ratio": s.get("comp_ratio"), "B_f": s.get("B_f"),
            "turn_budget": a.get("turn_budget"),
            "ctx_tokens": s.get("ctx_tokens"),
            "survived_turns": s.get("survived_turns"),
            "n_turns_requested": s.get("n_turns_requested"),
            "completed_all": s.get("completed_all"),
            "stop_reason": s.get("stop_reason"),
            "max_tokens_processed": s.get("max_tokens_processed"),
            "turn_ratio": s.get("turn_ratio"), "turn_cap": s.get("turn_cap"),
            "n_turn_reprunes": s.get("n_turn_reprunes"),
            "mean_acc": s.get("mean_acc"),
            "prefill_peak_gb": s.get("prefill_peak_gb"),
            "peak_during_turns_gb": s.get("peak_during_turns_gb"),
            "final_cache_gb": s.get("final_cache_gb"),
            "cache_gb_turn0": growth["c0"], "cache_gb_last": growth["clast"],
            "cache_growth_gb_per_turn": growth["slope"],
        })

    order = {"kvzip": 0, "full": 1, "streamllm": 2, "ours": 3, "ours_predict": 4, "ours_combine": 5}
    md = ["# SCBench multi-turn summary",
          "",
          "Multi-turn KV growth, OOM/stop, and accuracy. Better = more turns/tokens sustained at "
          "BOUNDED, slowly-growing cache with retained accuracy. `stop`: completed / oom / "
          "acc_collapse (5 consecutive 0-acc turns). `cache GB/turn` = mean per-turn cache growth.",
          ""]
    for (model, data), methods in sorted(groups.items()):
        md += [f"## {model} / {data}", ""]
        headers = ["method", "ctx_tok", "survived/req", "stop", "max_tok", "mean_acc",
                   "prefill_GB", "peak_turns_GB", "cacheGB turn0→last", "cache GB/turn"]
        rows = []
        for mk in sorted(methods, key=lambda k: order.get(k, 9)):
            s, g = methods[mk]
            rows.append([
                mk, s.get("ctx_tokens"),
                f"{s.get('survived_turns')}/{s.get('n_turns_requested')}",
                s.get("stop_reason"), s.get("max_tokens_processed"),
                fmt(s.get("mean_acc"), 3),
                fmt(s.get("prefill_peak_gb")), fmt(s.get("peak_during_turns_gb")),
                f"{fmt(g['c0'])}→{fmt(g['clast'])}", fmt(g["slope"], 5),
            ])
        md += [md_table(headers, rows), ""]
    return "\n".join(md), long_rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    a = p.parse_args()

    niah_md, niah_rows = aggregate_niah(a.results)
    if niah_md is not None:
        open(os.path.join(a.results, "summary_niah.md"), "w").write(niah_md)
        write_csv(os.path.join(a.results, "summary_niah.csv"), niah_rows)
        print(f"wrote {a.results}/summary_niah.md ({len(niah_rows)} rows)")
    else:
        print("no NIAH runs found")

    grid_md, grid_rows = aggregate_niah_grid(a.results)
    if grid_md is not None:
        open(os.path.join(a.results, "summary_niah_grid.md"), "w").write(grid_md)
        write_csv(os.path.join(a.results, "summary_niah_grid.csv"), grid_rows)
        print(f"wrote {a.results}/summary_niah_grid.md ({len(grid_rows)} rows)")
    else:
        print("no NIAH grid runs found")

    sc_md, sc_rows = aggregate_scbench(a.results)
    if sc_md is not None:
        open(os.path.join(a.results, "summary_scbench.md"), "w").write(sc_md)
        write_csv(os.path.join(a.results, "summary_scbench.csv"), sc_rows)
        print(f"wrote {a.results}/summary_scbench.md ({len(sc_rows)} rows)")
    else:
        print("no SCBench runs found")


if __name__ == "__main__":
    main()
