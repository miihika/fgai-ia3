"""Compute corpus BLEU/chrF from saved, aligned test predictions."""
import csv
import json

import sacrebleu
import matplotlib.pyplot as plt

from common import DATA, RESULTS, read_jsonl

TEST = read_jsonl(DATA / "test.jsonl")
MODELS = ("transformer", "mt5", "llama", "gemma")


def main():
    summary = []
    for name in MODELS:
        pred_path = RESULTS / f"{name}_predictions.jsonl"
        meta_path = RESULTS / f"{name}_run.json"
        if not pred_path.exists() or not meta_path.exists():
            summary.append({"model": name, "status": "not run"})
            continue
        predictions = read_jsonl(pred_path)
        if len(predictions) != len(TEST):
            raise ValueError(f"{name}: {len(predictions)} predictions but {len(TEST)} test examples")
        if any(p["source"] != t["source"] or p["reference"] != t["reference"]
               for p, t in zip(predictions, TEST)):
            raise ValueError(f"{name}: test set alignment mismatch")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        candidates = [r["prediction"] for r in predictions]
        references = [r["reference"] for r in predictions]
        summary.append({"model": name, "status": "complete", "checkpoint": meta["model"],
                        "method": meta["method"], "bleu": round(sacrebleu.corpus_bleu(candidates, [references]).score, 2),
                        "chrf": round(sacrebleu.corpus_chrf(candidates, [references]).score, 2),
                        "train_seconds": meta["train_seconds"], "inference_seconds": meta["inference_seconds"],
                        "inference_seconds_per_example": round(meta["inference_seconds"] / meta.get("timed_test_count", len(predictions)), 3),
                        "parameters": meta["parameters"], "test_count": len(predictions),
                        "timed_test_count": meta.get("timed_test_count", len(predictions))})
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cols = ("model", "status", "checkpoint", "method", "bleu", "chrf", "train_seconds",
            "inference_seconds", "inference_seconds_per_example", "parameters", "test_count", "timed_test_count")
    with (RESULTS / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(summary)
    with (RESULTS / "samples.md").open("w", encoding="utf-8") as f:
        f.write("# Sample translations\n\nGenerated from the shared held-out test set.\n")
        for i in range(min(5, len(TEST))):
            f.write(f"\n## Example {i+1}\n\n**English:** {TEST[i]['source']}\n\n**Reference:** {TEST[i]['reference']}\n\n")
            for name in MODELS:
                path = RESULTS / f"{name}_predictions.jsonl"
                if path.exists():
                    f.write(f"**{name}:** {read_jsonl(path)[i]['prediction']}\n\n")
    complete = [r for r in summary if r["status"] == "complete"]
    if complete:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, key, label in ((axes[0], "bleu", "BLEU"), (axes[1], "chrf", "chrF")):
            ax.bar([r["model"] for r in complete], [r[key] for r in complete], color="#3478a6")
            ax.set_ylabel(label)
            ax.set_title(f"Test set {label}")
            ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(RESULTS / "scores.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        names = [r["model"] for r in complete]
        axes[0].bar(names, [r["parameters"] / 1e6 for r in complete], color="#629c8a")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Parameters, millions (log scale)")
        axes[0].set_title("Model size")
        axes[1].bar(names, [r["inference_seconds_per_example"] for r in complete], color="#9d7399")
        axes[1].set_ylabel("Seconds per test example")
        axes[1].set_title("CPU inference latency")
        for ax in axes:
            ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(RESULTS / "compute.png", dpi=160)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
