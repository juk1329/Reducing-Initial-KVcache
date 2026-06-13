# NIAH length-ladder summary

Peak GPU memory (GB) during **prefill** vs context length. `OOM` = ran out of memory; blank = not run. Lower peak / further reach is better.

## llama3.2-3b  (GPU ~23.659GB, KV 0.11469 GB/1k tok)

### Prefill peak memory (GB)
| method \ ctx_len | 8000 | 16000 | 32000 | 64000 | 96000 | 120000 |
| --- | --- | --- | --- | --- | --- | --- |
| ours/per_token(ib=4096,wm=8192,fb=8192) | 11.28 | 11.78 | 12.25 | 12.72 | 12.72 | 12.72 |
| kvzip/per_token(r=0.3) | 8.46 | 9.38 | 11.22 | 14.91 | OOM |  |
| ours/per_token(cr=0.0) | 11.28 | 11.78 | 11.81 | 11.81 | 11.81 | 11.81 |
| kvzip/per_token(cr=0.0) | 8.46 | 9.38 | 11.22 | 14.91 | OOM |  |
| ours/per_token(cr=0.2) | 11.28 | 11.78 | 11.97 | 12.35 | 12.82 | 13.67 |
| kvzip/per_token(cr=0.2) | 8.46 | 9.38 | 11.22 | 14.91 | OOM |  |
| ours/per_token(cr=0.4) | 11.28 | 11.78 | 12.16 | 13.29 | 14.04 | 15.08 |
| kvzip/per_token(cr=0.4) | 8.46 | 9.38 | 11.22 | 14.91 | OOM |  |
| ours/per_token(cr=0.6) | 11.28 | 11.78 | 12.34 | 14.04 | 15.84 | 17.18 |
| kvzip/per_token(cr=0.6) | 8.46 | 9.38 | 11.22 | 14.91 | OOM |  |
| ours/per_token(cr=0.8) | 11.28 | 11.78 | 12.71 | 14.80 | 17.35 | OOM |
| kvzip/per_token(cr=0.8) | 8.46 | 9.38 | 11.22 | 14.91 | OOM |  |
| ours/per_token(cr=1.0) | 11.28 | 12.20 | 14.04 | 17.73 | OOM |  |
| kvzip/per_token(cr=1.0) | 8.46 | 9.38 | 11.22 | 14.91 | OOM |  |

### Needle accuracy (0-1)
| method \ ctx_len | 8000 | 16000 | 32000 | 64000 | 96000 | 120000 |
| --- | --- | --- | --- | --- | --- | --- |
| ours/per_token(ib=4096,wm=8192,fb=8192) | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| kvzip/per_token(r=0.3) | 0.00 | 0.00 | 0.00 | 0.00 |  |  |
| ours/per_token(cr=0.0) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| kvzip/per_token(cr=0.0) | 0.00 | 0.00 | 0.00 | 0.00 |  |  |
| ours/per_token(cr=0.2) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| kvzip/per_token(cr=0.2) | 1.00 | 0.00 | 0.00 | 0.00 |  |  |
| ours/per_token(cr=0.4) | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| kvzip/per_token(cr=0.4) | 0.00 | 0.00 | 0.00 | 0.00 |  |  |
| ours/per_token(cr=0.6) | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| kvzip/per_token(cr=0.6) | 1.00 | 1.00 | 0.00 | 0.00 |  |  |
| ours/per_token(cr=0.8) | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |  |
| kvzip/per_token(cr=0.8) | 1.00 | 1.00 | 0.00 | 0.00 |  |  |
| ours/per_token(cr=1.0) | 1.00 | 1.00 | 0.00 | 0.00 |  |  |
| kvzip/per_token(cr=1.0) | 1.00 | 1.00 | 0.00 | 0.00 |  |  |

### Headline: how far before OOM
| method | max_ok_ctx | max_ok_tok | peak_gb@max | acc@max | segments@max | first_oom_ctx |
| --- | --- | --- | --- | --- | --- | --- |
| ours/per_token(ib=4096,wm=8192,fb=8192) | 120000 | 119999 | 12.72 | 0.00 | 11 |  |
| kvzip/per_token(r=0.3) | 64000 | 63999 | 14.91 | 0.00 |  | 96000 |
| ours/per_token(cr=0.0) | 120000 | 119999 | 11.81 | 0.00 | 10 |  |
| kvzip/per_token(cr=0.0) | 64000 | 63999 | 14.91 | 0.00 |  | 96000 |
| ours/per_token(cr=0.2) | 120000 | 119999 | 13.67 | 0.00 | 11 |  |
| kvzip/per_token(cr=0.2) | 64000 | 63999 | 14.91 | 0.00 |  | 96000 |
| ours/per_token(cr=0.4) | 120000 | 119999 | 15.08 | 0.00 | 11 |  |
| kvzip/per_token(cr=0.4) | 64000 | 63999 | 14.91 | 0.00 |  | 96000 |
| ours/per_token(cr=0.6) | 120000 | 119999 | 17.18 | 0.00 | 11 |  |
| kvzip/per_token(cr=0.6) | 64000 | 63999 | 14.91 | 0.00 |  | 96000 |
| ours/per_token(cr=0.8) | 96000 | 95999 | 17.35 | 0.00 | 9 | 120000 |
| kvzip/per_token(cr=0.8) | 64000 | 63999 | 14.91 | 0.00 |  | 96000 |
| ours/per_token(cr=1.0) | 64000 | 63999 | 17.73 | 0.00 | 0 | 96000 |
| kvzip/per_token(cr=1.0) | 64000 | 63999 | 14.91 | 0.00 |  | 96000 |

## qwen3-1.7b  (GPU ~23.659GB, KV 0.11469 GB/1k tok)

### Prefill peak memory (GB)
| method \ ctx_len | 2000 | 4000 |
| --- | --- | --- |
| ours/per_token(ib=512,wm=1024,fb=1024) | 3.74 | 3.82 |
| streamllm/per_token(ib=512,wm=1024,fb=1024) |  | 3.72 |
| full/per_token |  | 3.94 |
| ours/per_head(ib=512,wm=1024,fb=1024) |  | 3.82 |
| ours/per_token(cr=0.4) | 3.73 | 3.82 |
| kvzip/per_token(cr=0.4) | 4.50 | 4.73 |

### Needle accuracy (0-1)
| method \ ctx_len | 2000 | 4000 |
| --- | --- | --- |
| ours/per_token(ib=512,wm=1024,fb=1024) | 0.50 | 0.50 |
| streamllm/per_token(ib=512,wm=1024,fb=1024) |  | 0.00 |
| full/per_token |  | 1.00 |
| ours/per_head(ib=512,wm=1024,fb=1024) |  | 0.50 |
| ours/per_token(cr=0.4) | 0.50 | 1.00 |
| kvzip/per_token(cr=0.4) | 1.00 | 1.00 |

### Headline: how far before OOM
| method | max_ok_ctx | max_ok_tok | peak_gb@max | acc@max | segments@max | first_oom_ctx |
| --- | --- | --- | --- | --- | --- | --- |
| ours/per_token(ib=512,wm=1024,fb=1024) | 4000 | 3999 | 3.82 | 0.50 | 3 |  |
| streamllm/per_token(ib=512,wm=1024,fb=1024) | 4000 | 3999 | 3.72 | 0.00 | 3 |  |
| full/per_token | 4000 | 3999 | 3.94 | 1.00 |  |  |
| ours/per_head(ib=512,wm=1024,fb=1024) | 4000 | 3999 | 3.82 | 0.50 | 3 |  |
| ours/per_token(cr=0.4) | 4000 | 3999 | 3.82 | 1.00 | 3 |  |
| kvzip/per_token(cr=0.4) | 4000 | 3999 | 4.73 | 1.00 |  |  |

## qwen3-4b  (GPU ~23.659GB, KV 0.11469 GB/1k tok)

### Prefill peak memory (GB)
| method \ ctx_len | 8000 | 16000 | 32000 | 64000 | 96000 | 120000 |
| --- | --- | --- | --- | --- | --- | --- |
| ours/per_token(ib=4096,wm=8192,fb=8192) | 14.37 | 14.95 | 15.64 | 15.64 | 15.64 | 15.64 |
| kvzip/per_token(r=0.3) | 10.67 | 11.85 | 14.22 | 18.96 | OOM |  |
| ours/per_token(cr=0.0) | 14.37 | 14.42 | 14.46 | 14.47 | 15.07 | 15.07 |
| kvzip/per_token(cr=0.0) | 10.67 | 11.85 | 14.22 | 18.96 | OOM |  |
| ours/per_token(cr=0.2) | 14.37 | 14.58 | 15.01 | 15.88 | 16.85 | 17.58 |
| kvzip/per_token(cr=0.2) | 10.67 | 11.85 | 14.22 | 18.96 | OOM |  |
| ours/per_token(cr=0.4) | 14.37 | 14.82 | 15.74 | 17.56 | 19.39 | OOM |
| kvzip/per_token(cr=0.4) | 10.67 | 11.85 | 14.22 | 18.96 | OOM |  |
| ours/per_token(cr=0.6) | 14.37 | 15.07 | 16.47 | 19.26 | OOM |  |
| kvzip/per_token(cr=0.6) | 10.67 | 11.85 | 14.22 | 18.96 | OOM |  |
| ours/per_token(cr=0.8) | 14.37 | 15.31 | 17.19 | 20.96 | OOM |  |
| kvzip/per_token(cr=0.8) | 10.67 | 11.85 | 14.22 | 18.96 | OOM |  |
| ours/per_token(cr=1.0) | 14.37 | 15.58 | 17.94 | OOM |  |  |
| kvzip/per_token(cr=1.0) | 10.67 | 11.85 | 14.22 | 18.96 | OOM |  |

### Needle accuracy (0-1)
| method \ ctx_len | 8000 | 16000 | 32000 | 64000 | 96000 | 120000 |
| --- | --- | --- | --- | --- | --- | --- |
| ours/per_token(ib=4096,wm=8192,fb=8192) | 1.00 | 1.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| kvzip/per_token(r=0.3) | 1.00 | 1.00 | 1.00 | 1.00 |  |  |
| ours/per_token(cr=0.0) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| kvzip/per_token(cr=0.0) | 0.00 | 0.00 | 0.00 | 0.00 |  |  |
| ours/per_token(cr=0.2) | 0.50 | 0.50 | 0.50 | 0.00 | 1.00 | 0.50 |
| kvzip/per_token(cr=0.2) | 1.00 | 1.00 | 1.00 | 1.00 |  |  |
| ours/per_token(cr=0.4) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| kvzip/per_token(cr=0.4) | 1.00 | 1.00 | 1.00 | 1.00 |  |  |
| ours/per_token(cr=0.6) | 1.00 | 1.00 | 1.00 | 1.00 |  |  |
| kvzip/per_token(cr=0.6) | 1.00 | 1.00 | 1.00 | 1.00 |  |  |
| ours/per_token(cr=0.8) | 1.00 | 1.00 | 1.00 | 1.00 |  |  |
| kvzip/per_token(cr=0.8) | 1.00 | 1.00 | 1.00 | 1.00 |  |  |
| ours/per_token(cr=1.0) | 1.00 | 1.00 | 1.00 |  |  |  |
| kvzip/per_token(cr=1.0) | 1.00 | 1.00 | 1.00 | 1.00 |  |  |

### Headline: how far before OOM
| method | max_ok_ctx | max_ok_tok | peak_gb@max | acc@max | segments@max | first_oom_ctx |
| --- | --- | --- | --- | --- | --- | --- |
| ours/per_token(ib=4096,wm=8192,fb=8192) | 120000 | 119999 | 15.64 | 0.00 | 14 |  |
| kvzip/per_token(r=0.3) | 64000 | 63999 | 18.96 | 1.00 |  | 96000 |
| ours/per_token(cr=0.0) | 120000 | 119999 | 15.07 | 0.00 | 14 |  |
| kvzip/per_token(cr=0.0) | 64000 | 63999 | 18.96 | 0.00 |  | 96000 |
| ours/per_token(cr=0.2) | 120000 | 119999 | 17.58 | 0.50 | 14 |  |
| kvzip/per_token(cr=0.2) | 64000 | 63999 | 18.96 | 1.00 |  | 96000 |
| ours/per_token(cr=0.4) | 96000 | 95999 | 19.39 | 1.00 | 11 | 120000 |
| kvzip/per_token(cr=0.4) | 64000 | 63999 | 18.96 | 1.00 |  | 96000 |
| ours/per_token(cr=0.6) | 64000 | 63999 | 19.26 | 1.00 | 7 | 96000 |
| kvzip/per_token(cr=0.6) | 64000 | 63999 | 18.96 | 1.00 |  | 96000 |
| ours/per_token(cr=0.8) | 64000 | 63999 | 20.96 | 1.00 | 7 | 96000 |
| kvzip/per_token(cr=0.8) | 64000 | 63999 | 18.96 | 1.00 |  | 96000 |
| ours/per_token(cr=1.0) | 32000 | 31999 | 17.94 | 1.00 | 0 | 64000 |
| kvzip/per_token(cr=1.0) | 64000 | 63999 | 18.96 | 1.00 |  | 96000 |
