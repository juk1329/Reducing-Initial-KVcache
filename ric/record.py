# ------------------------------------------------------------------------------
# Experiment result recording.
#
# Goal: every run is logged accurately and in full detail so paper.tex can be
# written from the recorded numbers (no need to re-run). Each run writes:
#   * results/<experiment>/<descriptive-timestamped>.json   (full detail)
#   * results/<experiment>_runs.jsonl  (one flat line per run, for aggregation)
# Use aggregate.py to turn these into paper-ready tables.
# ------------------------------------------------------------------------------
import json
import os
import socket
import time

import torch


def capture_env(model=None, device=0):
    """Snapshot the environment + model shape so results are self-describing.
    Includes KV bytes/token, which lets the paper compute theoretical memory."""
    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unix_time": time.time(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
    }
    try:
        env["gpu_name"] = torch.cuda.get_device_name(device)
        _, total = torch.cuda.mem_get_info(device)
        env["gpu_total_gb"] = round(total / 1e9, 3)
    except Exception:
        env["gpu_name"], env["gpu_total_gb"] = None, None
    for mod in ("transformers", "flash_attn"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None

    if model is not None:
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        try:
            dtype_bytes = torch.finfo(model.dtype).bits // 8
        except Exception:
            dtype_bytes = 2
        env["model_name"] = getattr(model, "name", None)
        env["model_config"] = {
            "num_hidden_layers": int(n_layers),
            "num_attention_heads": int(cfg.num_attention_heads),
            "num_key_value_heads": int(n_kv),
            "head_dim": int(head_dim),
            "hidden_size": int(cfg.hidden_size),
            "dtype": str(model.dtype),
        }
        kv_bytes = int(n_layers * n_kv * head_dim * 2 * dtype_bytes)  # K+V
        env["kv_bytes_per_token"] = kv_bytes
        env["kv_gb_per_1k_tokens"] = round(kv_bytes * 1000 / 1e9, 5)
    return env


def save_run(experiment: str, record: dict, out_dir: str = "results") -> str:
    """Write the detailed record JSON (timestamped, never overwritten) and append
    a flat summary line to the master JSONL. Returns the detailed file path."""
    exp_dir = os.path.join(out_dir, experiment)
    os.makedirs(exp_dir, exist_ok=True)

    args = record.get("args", {}) or {}
    parts = [
        str(args.get("model", "model")),
        str(args.get("method", "method")),
        str(args.get("level", "")),
        str(args.get("data", "")),
        str(args.get("tag", "") or ""),
        time.strftime("%Y%m%d-%H%M%S"),
    ]
    stem = "_".join(p for p in parts if p)
    path = os.path.join(exp_dir, stem + ".json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    master = os.path.join(out_dir, f"{experiment}_runs.jsonl")
    flat = {"file": path, "timestamp": record.get("timestamp")}
    flat.update({f"arg_{k}": v for k, v in args.items()})
    flat.update({f"sum_{k}": v for k, v in (record.get("summary") or {}).items()})
    env = record.get("env") or {}
    flat["gpu"] = env.get("gpu_name")
    flat["gpu_total_gb"] = env.get("gpu_total_gb")
    with open(master, "a") as f:
        f.write(json.dumps(flat, default=str) + "\n")

    print(f"\n[recorded] detail -> {path}\n[recorded] master -> {master}")
    return path
