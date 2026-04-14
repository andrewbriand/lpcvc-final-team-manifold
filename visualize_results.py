"""
Generate comparison charts from per-model result CSVs.

Usage:
    python visualize_results.py                          # auto-discover result CSVs in results/
    python visualize_results.py results/vit_b16_all.csv results/mobileclip2_b_all.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "results"
CHART_DIR = "results/charts"

ACCENT_METRICS = {
    "R@1": "Recall@1",
    "R@5": "Recall@5",
    "R@10": "Recall@10",
    "accuracy": "Binary Accuracy",
}

LATENCY_METRICS = {"img_ms", "txt_ms"}

DATASET_ORDER = [
    "sample",
    "mscoco",
    "flickr30k",
    "sugarcrepe",
    "sugarcrepe_pp_swap_att",
    "sugarcrepe_pp_replace_attribute",
    "sugarcrepe_pp_replace_object",
    "sugarcrepe_pp_replace_relation",
    "sugarcrepe_pp_swap_object",
]
DATASET_LABELS = {
    "sample": "Sample (56 imgs)",
    "mscoco": "MSCOCO 5K",
    "flickr30k": "Flickr30K",
    "sugarcrepe": "SugarCrepe",
    "sugarcrepe_pp_swap_att": "SugarCrepe++ Swap Attr",
    "sugarcrepe_pp_replace_attribute": "SugarCrepe++ Repl Attr",
    "sugarcrepe_pp_replace_object": "SugarCrepe++ Repl Obj",
    "sugarcrepe_pp_replace_relation": "SugarCrepe++ Repl Rel",
    "sugarcrepe_pp_swap_object": "SugarCrepe++ Swap Obj",
}


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


LEGACY_CSV = os.path.join(RESULTS_DIR, "20260228_042036.csv")


def discover_csvs() -> list[str]:
    paths = []
    if not os.path.isdir(RESULTS_DIR):
        return paths
    if os.path.isfile(LEGACY_CSV):
        paths.append(LEGACY_CSV)
    for name in sorted(os.listdir(RESULTS_DIR)):
        full = os.path.join(RESULTS_DIR, name)
        if full == LEGACY_CSV:
            continue
        if os.path.isfile(full) and name.endswith(".csv"):
            paths.append(full)
    return paths


def ordered_datasets(data: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    models = sorted(data.keys())
    available = {dataset for model in models for dataset in data[model].keys()}
    known = [dataset for dataset in DATASET_ORDER if dataset in available]
    extras = sorted(available - set(DATASET_ORDER))
    return known + extras


def _ingest_long_format(rows: list[dict], data: dict):
    """New-style CSVs: columns = timestamp,model,dataset,metric,value,contract_version."""
    for row in rows:
        model = row.get("model", "")
        dataset = row.get("dataset", "")
        metric = row.get("metric", "")
        raw = row.get("value", "")
        if not model or not dataset or not metric or metric == "error":
            continue
        try:
            data[model][dataset][metric] = float(raw)
        except (ValueError, TypeError):
            pass


def _ingest_wide_format(rows: list[dict], data: dict):
    """Legacy CSVs: columns = model,dataset,R@1,R@5,R@10,img_ms,txt_ms,accuracy,..."""
    wide_metrics = ["R@1", "R@5", "R@10", "img_ms", "txt_ms", "accuracy"]
    for row in rows:
        model = row.get("model", "")
        dataset = row.get("dataset", "")
        if not model or not dataset:
            continue
        for m in wide_metrics:
            raw = row.get(m, "")
            if not raw:
                continue
            try:
                data[model][dataset][m] = float(raw)
            except (ValueError, TypeError):
                pass


def merge_rows(csv_paths: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """model -> dataset -> metric -> value (latest file wins)."""
    data: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in csv_paths:
        rows = load_csv(path)
        if not rows:
            continue
        if "metric" in rows[0] and "value" in rows[0]:
            _ingest_long_format(rows, data)
        else:
            _ingest_wide_format(rows, data)
    return data


def print_table(data: dict) -> str:
    models = sorted(data.keys())
    datasets = ordered_datasets(data)

    lines: list[str] = []
    header = f"{'Model':<20}"
    for ds in datasets:
        header += f" | {DATASET_LABELS.get(ds, ds):>18}"
    lines.append(header)
    lines.append("-" * len(header))

    for metric_key, metric_label in ACCENT_METRICS.items():
        relevant_ds = [ds for ds in datasets if any(metric_key in data[m].get(ds, {}) for m in models)]
        if not relevant_ds:
            continue
        lines.append(f"\n  {metric_label}")
        for model in models:
            row = f"  {model:<18}"
            for ds in relevant_ds:
                val = data[model].get(ds, {}).get(metric_key)
                row += f" | {val:>18.4f}" if val is not None else f" | {'—':>18}"
            lines.append(row)

    lines.append(f"\n  Image latency (ms/img)")
    for model in models:
        row = f"  {model:<18}"
        for ds in datasets:
            val = data[model].get(ds, {}).get("img_ms")
            row += f" | {val:>18.2f}" if val is not None else f" | {'—':>18}"
        lines.append(row)

    return "\n".join(lines)


def plot_grouped_bar(data: dict, metric: str, title: str, ylabel: str, out_path: str):
    models = sorted(data.keys())
    datasets = [d for d in ordered_datasets(data) if any(metric in data[m].get(d, {}) for m in models)]
    if not datasets:
        return

    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 2.5), 5))

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(models), 1)))
    for i, model in enumerate(models):
        vals = [data[model].get(ds, {}).get(metric, 0) for ds in datasets]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=model, color=colors[i])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(max(bar.get_height() for bar in ax.patches) * 1.15, 1.05) if ax.patches else 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_latency(data: dict, out_path: str):
    models = sorted(data.keys())
    datasets = [d for d in ordered_datasets(data) if any("img_ms" in data[m].get(d, {}) for m in models)]
    if not datasets:
        return

    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 2.5), 5))

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(models), 1)))
    for i, model in enumerate(models):
        vals = [data[model].get(ds, {}).get("img_ms", 0) for ds in datasets]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=model, color=colors[i])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Image Latency (ms/image)")
    ax.set_title("Image Encoder Latency by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_radar(data: dict, out_path: str):
    models = sorted(data.keys())
    metrics_ds = [
        ("R@10", "mscoco", "COCO R@10"),
        ("R@10", "flickr30k", "Flickr R@10"),
        ("R@1", "flickr30k", "Flickr R@1"),
        ("accuracy", "sugarcrepe", "SugarCrepe Acc"),
    ]
    available = [(m, d, l) for m, d, l in metrics_ds if any(m in data[mdl].get(d, {}) for mdl in models)]
    if len(available) < 3:
        return

    labels = [l for _, _, l in available]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(models), 1)))
    for i, model in enumerate(models):
        vals = [data[model].get(d, {}).get(m, 0) for m, d, _ in available]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.5, label=model, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison (key metrics)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize model comparison results.")
    parser.add_argument("csvs", nargs="*", help="CSV files to include (default: auto-discover)")
    args = parser.parse_args()

    csv_paths = args.csvs or discover_csvs()
    if not csv_paths:
        print("No result CSVs found. Run validate.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(csv_paths)} result file(s):")
    for p in csv_paths:
        print(f"  {p}")

    data = merge_rows(csv_paths)
    if not data:
        print("No valid data found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(print_table(data))
    print(f"\n{'='*60}\n")

    os.makedirs(CHART_DIR, exist_ok=True)
    print("Generating charts:")

    plot_grouped_bar(data, "R@10", "Recall@10 by Dataset", "Recall@10", os.path.join(CHART_DIR, "recall_at_10.png"))
    plot_grouped_bar(data, "R@1", "Recall@1 by Dataset", "Recall@1", os.path.join(CHART_DIR, "recall_at_1.png"))
    plot_grouped_bar(data, "accuracy", "Binary Accuracy (SugarCrepe)", "Accuracy", os.path.join(CHART_DIR, "sugarcrepe_accuracy.png"))
    plot_latency(data, os.path.join(CHART_DIR, "latency.png"))
    plot_radar(data, os.path.join(CHART_DIR, "radar.png"))

    print(f"\nAll charts saved to {CHART_DIR}/")


if __name__ == "__main__":
    main()
