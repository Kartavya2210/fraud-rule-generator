"""Ablation study: how much do few-shot examples, schema-constrained decoding
and validation-feedback retries each contribute?

Each configuration generates a rule for every benchmark description, then
back-tests whatever valid rule comes out. Results land in eval/results/.

    python eval/run_eval.py --models qwen2.5:7b qwen2.5:1.5b
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest import backtest  # noqa: E402
from app.generator import GenerationError, generate_rule  # noqa: E402

HERE = Path(__file__).resolve().parent

CONFIGS = {
    "zero_shot_json":    dict(few_shot=False, constrained=False, max_attempts=1),
    "few_shot_json":     dict(few_shot=True,  constrained=False, max_attempts=1),
    "few_shot_schema":   dict(few_shot=True,  constrained=True,  max_attempts=1),
    "full_with_retry":   dict(few_shot=True,  constrained=True,  max_attempts=3),
}


def run(provider: str, model: str, configs: list[str], benchmark: list[dict]) -> list[dict]:
    rows = []
    for cfg_name in configs:
        for case in benchmark:
            t0 = time.perf_counter()
            row = {"config": cfg_name, "model": model, "id": case["id"], "target": case.get("target")}
            try:
                gen = generate_rule(case["description"], provider=provider, model=model, **CONFIGS[cfg_name])
                row.update(valid=True, attempts=gen.attempts, first_try=gen.attempts == 1,
                           rule=gen.rule.model_dump(mode="json"), rejected=gen.errors)
                bt = backtest(gen.rule)
                m = bt["metrics"]
                row.update(backtest_passed=bt["passed"], precision=m["precision"], recall=m["recall"],
                           flagged=m["flagged"], engine_agreement=bt["engines"]["agreement"],
                           target_recall=bt["recall_by_fraud_type"].get(case.get("target"), {}).get("recall"))
            except GenerationError as e:
                row.update(valid=False, attempts=e.result.attempts, first_try=False, rejected=e.result.errors,
                           last_output=e.result.raw_outputs[-1][:1000] if e.result.raw_outputs else None)
            row["latency_s"] = round(time.perf_counter() - t0, 1)
            rows.append(row)
            status = "PASS" if row.get("backtest_passed") else ("valid" if row["valid"] else "INVALID")
            print(f"[{model} | {cfg_name:16s}] {case['id']:22s} {status:7s} "
                  f"attempts={row['attempts']} {row['latency_s']}s", flush=True)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["model"], r["config"]) for r in rows}, key=lambda k: (k[0], list(CONFIGS).index(k[1])))
    for model, cfg in keys:
        rs = [r for r in rows if r["model"] == model and r["config"] == cfg]
        valid = [r for r in rs if r["valid"]]
        targeted = [r for r in valid if r["target"] and r.get("target_recall") is not None]
        n = len(rs)
        out.append({
            "model": model, "config": cfg, "cases": n,
            "valid_first_try": round(sum(r["first_try"] and r["valid"] for r in rs) / n, 3),
            "valid_final": round(len(valid) / n, 3),
            "backtest_pass": round(sum(bool(r.get("backtest_passed")) for r in rs) / n, 3),
            "mean_precision": round(sum(r["precision"] for r in valid) / len(valid), 3) if valid else None,
            "mean_target_recall": round(sum(r["target_recall"] for r in targeted) / len(targeted), 3)
            if targeted else None,
            "engine_agreement": all(r.get("engine_agreement", True) for r in valid),
            "mean_latency_s": round(sum(r["latency_s"] for r in rs) / n, 1),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--models", nargs="+", default=["qwen2.5:7b"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    args = ap.parse_args()

    benchmark = json.loads((HERE / "benchmark.json").read_text())
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    rows = []
    for model in args.models:
        model_rows = run(args.provider, model, args.configs, benchmark)
        (results_dir / f"{args.provider}_{model.replace(':', '-')}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in model_rows))
        rows += model_rows

    summary = summarise(rows)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    cols = ["model", "config", "valid_first_try", "valid_final", "backtest_pass",
            "mean_precision", "mean_target_recall", "mean_latency_s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(s[c]) for c in cols) + " |" for s in summary]
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
