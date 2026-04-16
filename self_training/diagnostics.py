"""Diagnostic evaluation for retrieval models.

Goes beyond R@K to identify failure modes, blind spots, and
similarity distribution characteristics. Produces a structured
JSON report + human-readable summary.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def compute_retrieval_diagnostics(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    img2txt: list[list[int]],
    captions: list[str],
    image_ids: list[str] | None = None,
) -> dict:
    """Full diagnostic analysis of image-to-text retrieval.

    Returns a dict with:
      - aggregate: R@1/5/10, MedR, MeanR
      - similarity_stats: distributions for positive/negative pairs
      - per_query: list of per-image results (rank, scores, retrieved texts)
      - failure_analysis: categorized failure modes
      - hardest_queries: bottom-20 by rank of first correct match
      - easiest_queries: top-20 by rank
    """
    n_images = image_embeddings.shape[0]
    n_texts = text_embeddings.shape[0]

    # Full similarity matrix
    sim = image_embeddings @ text_embeddings.T  # (n_images, n_texts)

    # --- Per-query analysis ---
    per_query = []
    ranks_of_first_correct = []
    positive_sims = []
    negative_sims = []
    margin_list = []  # gap between best positive and best negative

    for i in range(n_images):
        gt_indices = set(img2txt[i])
        sorted_indices = np.argsort(-sim[i])
        rank_list = sorted_indices.tolist()

        # Find rank of first correct match
        first_correct_rank = n_texts  # worst case
        for r, idx in enumerate(rank_list):
            if idx in gt_indices:
                first_correct_rank = r + 1  # 1-indexed
                break

        ranks_of_first_correct.append(first_correct_rank)

        # Positive similarities (ground truth)
        pos_scores = [float(sim[i, g]) for g in gt_indices]
        positive_sims.extend(pos_scores)
        best_positive = max(pos_scores) if pos_scores else 0.0

        # Top-K retrieved (for analysis)
        top_k = 20
        top_indices = rank_list[:top_k]
        top_scores = [float(sim[i, j]) for j in top_indices]
        top_captions = [captions[j] for j in top_indices]
        top_is_correct = [j in gt_indices for j in top_indices]

        # Best negative similarity
        neg_scores_top = [s for s, c in zip(top_scores, top_is_correct) if not c]
        best_negative = max(neg_scores_top) if neg_scores_top else 0.0

        # Negative similarities (sample from non-GT)
        non_gt = [j for j in range(min(100, n_texts)) if j not in gt_indices]
        neg_sample = [float(sim[i, j]) for j in non_gt[:50]]
        negative_sims.extend(neg_sample)

        margin = best_positive - best_negative
        margin_list.append(margin)

        gt_captions = [captions[g] for g in gt_indices]

        query_result = {
            "query_idx": i,
            "query_id": image_ids[i] if image_ids else str(i),
            "first_correct_rank": first_correct_rank,
            "best_positive_sim": round(best_positive, 4),
            "best_negative_sim": round(best_negative, 4),
            "margin": round(margin, 4),
            "gt_captions": gt_captions[:3],  # truncate for readability
            "top5_retrieved": [
                {"rank": r + 1, "caption": c[:100], "sim": round(s, 4), "correct": cr}
                for r, (c, s, cr) in enumerate(zip(top_captions[:5], top_scores[:5], top_is_correct[:5]))
            ],
        }
        per_query.append(query_result)

    ranks = np.array(ranks_of_first_correct)
    pos_arr = np.array(positive_sims)
    neg_arr = np.array(negative_sims)
    margin_arr = np.array(margin_list)

    # --- Aggregate metrics ---
    aggregate = {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "R@20": float(np.mean(ranks <= 20)),
        "R@50": float(np.mean(ranks <= 50)),
        "MedR": int(np.median(ranks)),
        "MeanR": float(np.mean(ranks)),
        "n_queries": n_images,
        "n_candidates": n_texts,
    }

    # --- Similarity distributions ---
    similarity_stats = {
        "positive": {
            "mean": round(float(pos_arr.mean()), 4),
            "std": round(float(pos_arr.std()), 4),
            "min": round(float(pos_arr.min()), 4),
            "p25": round(float(np.percentile(pos_arr, 25)), 4),
            "median": round(float(np.median(pos_arr)), 4),
            "p75": round(float(np.percentile(pos_arr, 75)), 4),
            "max": round(float(pos_arr.max()), 4),
        },
        "negative": {
            "mean": round(float(neg_arr.mean()), 4),
            "std": round(float(neg_arr.std()), 4),
            "min": round(float(neg_arr.min()), 4),
            "p25": round(float(np.percentile(neg_arr, 25)), 4),
            "median": round(float(np.median(neg_arr)), 4),
            "p75": round(float(np.percentile(neg_arr, 75)), 4),
            "max": round(float(neg_arr.max()), 4),
        },
        "margin": {
            "mean": round(float(margin_arr.mean()), 4),
            "std": round(float(margin_arr.std()), 4),
            "min": round(float(margin_arr.min()), 4),
            "p10": round(float(np.percentile(margin_arr, 10)), 4),
            "median": round(float(np.median(margin_arr)), 4),
            "p90": round(float(np.percentile(margin_arr, 90)), 4),
            "max": round(float(margin_arr.max()), 4),
        },
        "overlap_rate": round(float(np.mean(margin_arr < 0)), 4),
    }

    # --- Failure analysis ---
    # Categorize by difficulty buckets
    failure_buckets = {
        "correct_at_1": int(np.sum(ranks == 1)),
        "correct_at_2_5": int(np.sum((ranks >= 2) & (ranks <= 5))),
        "correct_at_6_10": int(np.sum((ranks >= 6) & (ranks <= 10))),
        "correct_at_11_50": int(np.sum((ranks >= 11) & (ranks <= 50))),
        "correct_at_51_plus": int(np.sum(ranks > 50)),
    }

    # Negative margin queries — model prefers wrong caption over GT
    neg_margin_queries = [q for q in per_query if q["margin"] < 0]
    near_miss_queries = [q for q in per_query if 0 <= q["margin"] < 0.05]

    failure_analysis = {
        "rank_distribution": failure_buckets,
        "negative_margin_count": len(neg_margin_queries),
        "negative_margin_pct": round(len(neg_margin_queries) / max(n_images, 1) * 100, 1),
        "near_miss_count": len(near_miss_queries),
        "near_miss_pct": round(len(near_miss_queries) / max(n_images, 1) * 100, 1),
    }

    # --- Hardest / easiest queries ---
    sorted_by_rank = sorted(per_query, key=lambda x: -x["first_correct_rank"])
    hardest = sorted_by_rank[:20]
    easiest = sorted_by_rank[-20:]

    # --- Caption-level analysis ---
    # Which GT captions are never retrieved in top-10?
    caption_retrieved_count = defaultdict(int)
    caption_total_count = defaultdict(int)
    for i in range(n_images):
        gt_indices = set(img2txt[i])
        top10 = set(np.argsort(-sim[i])[:10].tolist())
        for g in gt_indices:
            caption_total_count[g] += 1
            if g in top10:
                caption_retrieved_count[g] += 1

    never_retrieved = []
    for cap_idx in caption_total_count:
        if caption_retrieved_count[cap_idx] == 0:
            never_retrieved.append({
                "caption_idx": int(cap_idx),
                "caption": captions[cap_idx][:120],
                "times_as_gt": caption_total_count[cap_idx],
            })
    never_retrieved.sort(key=lambda x: -x["times_as_gt"])

    caption_analysis = {
        "total_gt_captions": len(caption_total_count),
        "never_retrieved_in_top10": len(never_retrieved),
        "never_retrieved_pct": round(len(never_retrieved) / max(len(caption_total_count), 1) * 100, 1),
        "worst_captions": never_retrieved[:20],
    }

    return {
        "aggregate": aggregate,
        "similarity_stats": similarity_stats,
        "failure_analysis": failure_analysis,
        "caption_analysis": caption_analysis,
        "hardest_queries": hardest,
        "easiest_queries": easiest,
        "per_query": per_query,  # full list — large, write to file
    }


def format_diagnostic_summary(report: dict) -> str:
    """Human-readable summary from a diagnostic report."""
    lines = []
    agg = report["aggregate"]
    sim = report["similarity_stats"]
    fail = report["failure_analysis"]
    cap = report["caption_analysis"]

    lines.append("=" * 60)
    lines.append("RETRIEVAL DIAGNOSTIC SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Queries: {agg['n_queries']}  Candidates: {agg['n_candidates']}")
    lines.append(f"R@1: {agg['R@1']:.4f}  R@5: {agg['R@5']:.4f}  R@10: {agg['R@10']:.4f}  R@20: {agg['R@20']:.4f}")
    lines.append(f"MedR: {agg['MedR']}  MeanR: {agg['MeanR']:.1f}")
    lines.append("")

    lines.append("--- Similarity Distributions ---")
    lines.append(f"Positive pairs:  mean={sim['positive']['mean']:.4f}  std={sim['positive']['std']:.4f}  "
                 f"range=[{sim['positive']['min']:.4f}, {sim['positive']['max']:.4f}]")
    lines.append(f"Negative pairs:  mean={sim['negative']['mean']:.4f}  std={sim['negative']['std']:.4f}  "
                 f"range=[{sim['negative']['min']:.4f}, {sim['negative']['max']:.4f}]")
    lines.append(f"Margin (pos-neg): mean={sim['margin']['mean']:.4f}  std={sim['margin']['std']:.4f}  "
                 f"p10={sim['margin']['p10']:.4f}  p90={sim['margin']['p90']:.4f}")
    sep = sim["positive"]["mean"] - sim["negative"]["mean"]
    lines.append(f"Mean separation:  {sep:.4f}")
    lines.append(f"Overlap rate (margin < 0): {sim['overlap_rate']*100:.1f}%")
    lines.append("")

    lines.append("--- Rank Distribution ---")
    rd = fail["rank_distribution"]
    total = agg["n_queries"]
    lines.append(f"  Rank 1:       {rd['correct_at_1']:4d} ({rd['correct_at_1']/total*100:5.1f}%)")
    lines.append(f"  Rank 2-5:     {rd['correct_at_2_5']:4d} ({rd['correct_at_2_5']/total*100:5.1f}%)")
    lines.append(f"  Rank 6-10:    {rd['correct_at_6_10']:4d} ({rd['correct_at_6_10']/total*100:5.1f}%)")
    lines.append(f"  Rank 11-50:   {rd['correct_at_11_50']:4d} ({rd['correct_at_11_50']/total*100:5.1f}%)")
    lines.append(f"  Rank 51+:     {rd['correct_at_51_plus']:4d} ({rd['correct_at_51_plus']/total*100:5.1f}%)")
    lines.append("")

    lines.append("--- Failure Modes ---")
    lines.append(f"Negative margin (model prefers wrong caption): {fail['negative_margin_count']} ({fail['negative_margin_pct']}%)")
    lines.append(f"Near-miss (margin < 0.05):                     {fail['near_miss_count']} ({fail['near_miss_pct']}%)")
    lines.append("")

    lines.append("--- Caption Blind Spots ---")
    lines.append(f"GT captions never retrieved in top-10: {cap['never_retrieved_in_top10']}/{cap['total_gt_captions']} ({cap['never_retrieved_pct']}%)")
    if cap["worst_captions"]:
        lines.append("Worst blind-spot captions:")
        for c in cap["worst_captions"][:10]:
            lines.append(f"  [{c['times_as_gt']}x GT] \"{c['caption']}\"")
    lines.append("")

    lines.append("--- Hardest Queries (worst rank of first correct) ---")
    for q in report["hardest_queries"][:10]:
        lines.append(f"  Query {q['query_id']}: rank={q['first_correct_rank']}  "
                     f"margin={q['margin']:.4f}  GT: \"{q['gt_captions'][0][:80]}\"")
        if q["top5_retrieved"]:
            top1 = q["top5_retrieved"][0]
            lines.append(f"    Top-1 retrieved: \"{top1['caption'][:80]}\" (sim={top1['sim']:.4f})")
    lines.append("")

    return "\n".join(lines)


def run_diagnostics(
    model,
    preprocess,
    tokenizer,
    dataset_key: str,
    device: str = "cuda",
    batch_size: int = 64,
    hf_cache_dir: str = "hf_cache",
    img_csv: str = "dataset/img_list.csv",
    txt_csv: str = "dataset/txt_list.csv",
    img_dir: str = "dataset/images",
    output_dir: Path | None = None,
) -> dict:
    """Run full diagnostics on a retrieval dataset. Returns the report dict."""
    import torch
    from validate import (
        DATASETS,
        embed_images,
        embed_texts,
    )

    task_type, source, desc = DATASETS[dataset_key]
    if task_type != "retrieval":
        raise ValueError(f"Diagnostics only support retrieval datasets, got '{task_type}' for '{dataset_key}'")

    # Load dataset
    if dataset_key == "sample":
        from validate import load_sample
        images, captions, img2txt = load_sample(img_csv, txt_csv, img_dir)
    else:
        from validate import load_mscoco, load_flickr30k
        if dataset_key == "mscoco":
            images, captions, img2txt = load_mscoco(hf_cache_dir)
        elif dataset_key == "flickr30k":
            images, captions, img2txt = load_flickr30k(hf_cache_dir)
        else:
            raise ValueError(f"Unsupported retrieval dataset: {dataset_key}")

    print(f"Running diagnostics on {dataset_key}: {len(images)} images, {len(captions)} captions")

    # Embed
    with torch.no_grad():
        img_embs, _ = embed_images(model, preprocess, images, device, batch_size, cache=None)
        txt_embs, _ = embed_texts(model, tokenizer, captions, device, batch_size, cache=None)

    # Run analysis
    image_ids = [f"{dataset_key}_{i}" for i in range(len(images))]
    report = compute_retrieval_diagnostics(img_embs, txt_embs, img2txt, captions, image_ids)
    report["dataset"] = dataset_key

    # Print summary
    summary = format_diagnostic_summary(report)
    print(summary)

    # Save if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full report (without per_query which is huge)
        compact = {k: v for k, v in report.items() if k != "per_query"}
        with open(output_dir / f"diagnostics_{dataset_key}.json", "w") as f:
            json.dump(compact, f, indent=2)

        # Save per-query details separately
        with open(output_dir / f"per_query_{dataset_key}.json", "w") as f:
            json.dump(report["per_query"], f, indent=2)

        # Save human-readable summary
        with open(output_dir / f"summary_{dataset_key}.txt", "w") as f:
            f.write(summary)

        print(f"Reports saved to {output_dir}/")

    return report


def compare_reports(baseline_report: dict, trained_report: dict) -> str:
    """Compare two diagnostic reports and highlight changes."""
    lines = []
    lines.append("=" * 60)
    lines.append("COMPARISON: Baseline vs Self-Trained")
    lines.append("=" * 60)
    lines.append("")

    ba = baseline_report["aggregate"]
    ta = trained_report["aggregate"]

    lines.append(f"{'Metric':<12} {'Baseline':>10} {'Trained':>10} {'Delta':>10}")
    lines.append("-" * 44)
    for metric in ["R@1", "R@5", "R@10", "R@20", "MedR", "MeanR"]:
        bv = ba[metric]
        tv = ta[metric]
        delta = tv - bv
        sign = "+" if delta > 0 else ""
        if metric in ("MedR", "MeanR"):
            lines.append(f"{metric:<12} {bv:>10.1f} {tv:>10.1f} {sign}{delta:>9.1f}")
        else:
            lines.append(f"{metric:<12} {bv:>10.4f} {tv:>10.4f} {sign}{delta:>9.4f}")
    lines.append("")

    bs = baseline_report["similarity_stats"]
    ts = trained_report["similarity_stats"]
    lines.append("Similarity shift:")
    lines.append(f"  Positive mean: {bs['positive']['mean']:.4f} -> {ts['positive']['mean']:.4f}")
    lines.append(f"  Negative mean: {bs['negative']['mean']:.4f} -> {ts['negative']['mean']:.4f}")
    lines.append(f"  Margin mean:   {bs['margin']['mean']:.4f} -> {ts['margin']['mean']:.4f}")
    lines.append(f"  Overlap rate:  {bs['overlap_rate']*100:.1f}% -> {ts['overlap_rate']*100:.1f}%")
    lines.append("")

    bf = baseline_report["failure_analysis"]
    tf = trained_report["failure_analysis"]
    lines.append("Failure mode shift:")
    lines.append(f"  Neg-margin queries: {bf['negative_margin_count']} -> {tf['negative_margin_count']}")
    lines.append(f"  Near-miss queries:  {bf['near_miss_count']} -> {tf['near_miss_count']}")
    lines.append("")

    # Which queries got better/worse?
    bq = {q["query_id"]: q for q in baseline_report.get("per_query", [])}
    tq = {q["query_id"]: q for q in trained_report.get("per_query", [])}
    common = set(bq.keys()) & set(tq.keys())

    improved = []
    regressed = []
    for qid in common:
        br = bq[qid]["first_correct_rank"]
        tr = tq[qid]["first_correct_rank"]
        if tr < br:
            improved.append((qid, br, tr))
        elif tr > br:
            regressed.append((qid, br, tr))

    lines.append(f"Per-query changes ({len(common)} common queries):")
    lines.append(f"  Improved: {len(improved)} ({len(improved)/max(len(common),1)*100:.1f}%)")
    lines.append(f"  Regressed: {len(regressed)} ({len(regressed)/max(len(common),1)*100:.1f}%)")
    lines.append(f"  Unchanged: {len(common) - len(improved) - len(regressed)}")

    if regressed:
        regressed.sort(key=lambda x: x[2] - x[1], reverse=True)
        lines.append("")
        lines.append("  Worst regressions:")
        for qid, br, tr in regressed[:10]:
            lines.append(f"    {qid}: rank {br} -> {tr} (delta +{tr - br})")

    if improved:
        improved.sort(key=lambda x: x[1] - x[2], reverse=True)
        lines.append("")
        lines.append("  Best improvements:")
        for qid, br, tr in improved[:10]:
            lines.append(f"    {qid}: rank {br} -> {tr} (delta -{br - tr})")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Run retrieval diagnostics")
    parser.add_argument("--model", default="mobileclip2_s2")
    parser.add_argument("--dataset", default="sample", choices=["sample", "mscoco", "flickr30k"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnostics"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="LoRA checkpoint to evaluate")
    parser.add_argument("--compare-baseline", action="store_true", help="Run baseline too and compare")
    args = parser.parse_args()

    from validate import load_model, auto_device
    device = args.device or auto_device()

    if args.checkpoint:
        from self_training.merge_and_export import merge_lora_checkpoint
        model, preprocess, tokenizer = merge_lora_checkpoint(args.model, args.checkpoint, device=device)
    else:
        model, preprocess, tokenizer = load_model(args.model, device)

    report = run_diagnostics(
        model, preprocess, tokenizer,
        dataset_key=args.dataset,
        device=device,
        output_dir=args.output_dir,
    )

    if args.compare_baseline and args.checkpoint:
        print("\n\nRunning baseline for comparison...")
        base_model, base_pp, base_tok = load_model(args.model, device)
        baseline_report = run_diagnostics(
            base_model, base_pp, base_tok,
            dataset_key=args.dataset,
            device=device,
            output_dir=args.output_dir / "baseline",
        )
        comparison = compare_reports(baseline_report, report)
        print("\n" + comparison)
        with open(args.output_dir / f"comparison_{args.dataset}.txt", "w") as f:
            f.write(comparison)
