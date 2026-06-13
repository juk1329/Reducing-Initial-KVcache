# SCBench multi-turn summary

Multi-turn KV growth, OOM/stop, and accuracy. Better = more turns/tokens sustained at BOUNDED, slowly-growing cache with retained accuracy. `stop`: completed / oom / acc_collapse (5 consecutive 0-acc turns). `cache GB/turn` = mean per-turn cache growth.

## qwen3-4b / scbench_kv_short

| method | ctx_tok | survived/req | stop | max_tok | mean_acc | prefill_GB | peak_turns_GB | cacheGB turn0→last | cache GB/turn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kvzip | 24470 | 150/150 | completed | 43998 | 1.000 | 13.11 | 14.66 | 3.60→6.50 | 0.01946 |
| ours | 24470 | 150/150 | completed | 43998 | 1.000 | 13.17 | 13.10 | 3.60→4.80 | 0.00805 |
| ours_combine | 24470 | 150/150 | completed | 43998 | 1.000 | 13.17 | 13.10 | 3.60→4.80 | 0.00805 |
| ours_predict_v1 | 24470 | 150/150 | completed | 37915 | 0.907 | 13.17 | 12.67 | 3.60→4.40 | 0.00537 |
| ours_predict_v2 | 24470 | 150/150 | completed | 37901 | 0.907 | 13.17 | 12.67 | 3.60→4.40 | 0.00537 |

## qwen3-4b / scbench_summary_short

| method | ctx_tok | survived/req | stop | max_tok | mean_acc | prefill_GB | peak_turns_GB | cacheGB turn0→last | cache GB/turn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kvzip | 9634 | 100/100 | completed | 17678 | 0.487 | 10.91 | 9.86 | 0.60→1.80 | 0.01212 |
| ours | 9634 | 100/100 | completed | 17303 | 0.530 | 10.46 | 9.14 | 0.60→1.00 | 0.00404 |
| ours_combine | 9634 | 100/100 | completed | 16771 | 0.448 | 10.67 | 9.01 | 0.50→0.90 | 0.00404 |
| ours_predict_v1 | 9634 | 100/100 | completed | 17763 | 0.435 | 10.59 | 9.07 | 0.50→1.00 | 0.00505 |
| ours_predict_v2 | 9634 | 100/100 | completed | 17467 | 0.440 | 10.59 | 9.05 | 0.50→0.90 | 0.00404 |
