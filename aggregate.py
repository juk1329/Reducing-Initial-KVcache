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
        mkey = f"{a.get('method')}/{a.get('level')}"
        if a.get("method") == "kvzip":
            mkey += f"(r={a.get('ratio')})"
        elif a.get("method") in ("ours", "streamllm"):
            mkey += f"(bm={a.get('budget_max')},bt={a.get('budget_target')})"
        m = models.setdefault(model, {})
        m.setdefault(mkey, {})  # mkey -> {ctx_len: row}
        for row in r.get("rows", []):
            m[mkey][row["ctx_len"]] = row  # latest run wins
            long_rows.append({
                "model": model, "method": a.get("method"), "level": a.get("level"),
                "budget_max": a.get("budget_max"), "budget_target": a.get("budget_target"),
                "ratio": a.get("ratio"),
                "ctx_len": row["ctx_len"], "true_len": row.get("true_len"),
                "prefill_status": row.get("prefill_status"),
                "prefill_peak_gb": row.get("prefill_peak_gb"),
                "answer_peak_gb": row.get("answer_peak_gb"),
                "cache_gb": row.get("cache_gb"),
                "num_compactions": row.get("num_compactions"),
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
                best.get("num_compactions") if best else "",
                (min(r["ctx_len"] for r in ooms) if ooms else ""),
            ])
        md += ["### Headline: how far before OOM",
               md_table(["method", "max_ok_ctx", "max_ok_tok", "peak_gb@max",
                         "acc@max", "compactions@max", "first_oom_ctx"], hrows), ""]
    return "\n".join(md), long_rows


# -------------------------------------------------------------------- SCBench
def aggregate_scbench(results_dir):
    recs = load_records(os.path.join(results_dir, "scbench"))
    if not recs:
        return None, []
    groups = {}  # (model, data) -> {method: summary}
    long_rows = []
    for r in recs:
        a = r.get("args", {})
        model = a.get("model", "?")
        data = a.get("data", "?")
        s = r.get("summary", {}) or {}
        mkey = f"{a.get('method')}"
        if a.get("method") in ("ours", "streamllm"):
            mkey += f"(bm={a.get('budget_max')},bt={a.get('budget_target')},tb={a.get('turn_budget')})"
        groups.setdefault((model, data), {})[mkey] = s
        long_rows.append({
            "model": model, "data": data, "method": a.get("method"),
            "budget_max": a.get("budget_max"), "budget_target": a.get("budget_target"),
            "turn_budget": a.get("turn_budget"),
            "ctx_tokens": s.get("ctx_tokens"),
            "survived_turns": s.get("survived_turns"),
            "n_turns_requested": s.get("n_turns_requested"),
            "completed_all": s.get("completed_all"),
            "mean_acc": s.get("mean_acc"),
            "prefill_peak_gb": s.get("prefill_peak_gb"),
            "peak_during_turns_gb": s.get("peak_during_turns_gb"),
            "final_cache_gb": s.get("final_cache_gb"),
        })

    md = ["# SCBench short-context multi-turn summary",
          "",
          "Turns survived before OOM, accuracy, and memory. Surviving more turns at "
          "bounded memory with retained accuracy is better.",
          ""]
    for (model, data), methods in sorted(groups.items()):
        md += [f"## {model} / {data}", ""]
        headers = ["method", "ctx_tok", "survived/req", "mean_acc",
                   "prefill_peak_gb", "peak_turns_gb", "final_cache_gb"]
        rows = []
        for mk, s in methods.items():
            rows.append([
                mk, s.get("ctx_tokens"),
                f"{s.get('survived_turns')}/{s.get('n_turns_requested')}",
                fmt(s.get("mean_acc"), 3),
                fmt(s.get("prefill_peak_gb")), fmt(s.get("peak_during_turns_gb")),
                fmt(s.get("final_cache_gb")),
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

    sc_md, sc_rows = aggregate_scbench(a.results)
    if sc_md is not None:
        open(os.path.join(a.results, "summary_scbench.md"), "w").write(sc_md)
        write_csv(os.path.join(a.results, "summary_scbench.csv"), sc_rows)
        print(f"wrote {a.results}/summary_scbench.md ({len(sc_rows)} rows)")
    else:
        print("no SCBench runs found")


if __name__ == "__main__":
    main()
