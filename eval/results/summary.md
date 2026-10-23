| model | config | valid_first_try | valid_final | backtest_pass | mean_precision | mean_target_recall | adversarial_rejected | mean_latency_s |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:1.5b | zero_shot_json | 0.0 | 0.0 | 0.0 | None | None | 2/2 | 8.8 |
| qwen2.5:1.5b | few_shot_json | 0.818 | 0.818 | 0.545 | 0.597 | 0.333 | 2/2 | 4.3 |
| qwen2.5:1.5b | few_shot_schema | 0.909 | 0.909 | 0.636 | 0.637 | 0.348 | 2/2 | 4.3 |
| qwen2.5:1.5b | full_with_retry | 0.909 | 1.0 | 0.727 | 0.63 | 0.409 | 2/2 | 5.1 |
| qwen2.5:7b | zero_shot_json | 0.818 | 0.818 | 0.455 | 0.45 | 0.348 | 1/2 | 22.2 |
| qwen2.5:7b | few_shot_json | 1.0 | 1.0 | 0.818 | 0.719 | 0.603 | 2/2 | 22.5 |
| qwen2.5:7b | few_shot_schema | 1.0 | 1.0 | 0.818 | 0.719 | 0.603 | 2/2 | 20.3 |
| qwen2.5:7b | full_with_retry | 1.0 | 1.0 | 0.818 | 0.719 | 0.603 | 1/2 | 23.5 |
