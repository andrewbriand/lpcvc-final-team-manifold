"""Compute the kill-criterion threshold for FG-CLIP 2 retokenizer Run #1.

We run two checkpoints through validate.py's MS-COCO image-to-text retrieval
eval and record R@10:

  1. ``mobileclip2_s2`` — the model behind our 0.5839 hidden-leaderboard score.
  2. ``fgclip2_base_retokenized`` initialized with the FVT lookup table only
     (no training) — the 0.56-leaderboard hack equivalent.

Output: ``~/lpcvc/.cache/calibration.json`` with::

    {
      "baseline_COCO_R10":     <float>,
      "lookup_hack_COCO_R10":  <float>,
      "kill_at":               max(lookup_hack, baseline - 0.05),
      "calibrated_at":         "<ISO>"
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import validate as V  # noqa: E402
from lpcvc_models import MODELS  # noqa: E402


# ---------------------------------------------------------------------------
def _hashlib_token_sig(model_key: str, spec, ckpt_hash: str) -> str:
    import hashlib

    if spec.loader == "fgclip2_retokenized_hf":
        return hashlib.sha1(
            f"{model_key}|fgclip2_retokenized|openai_clip_bpe|{spec.text_max_length}|{ckpt_hash}".encode("utf-8")
        ).hexdigest()[:12]
    return hashlib.sha1(
        f"{model_key}|{spec.loader}|{spec.hf_repo}|{spec.hf_base_model}|{spec.text_max_length}".encode("utf-8")
    ).hexdigest()[:12]


def _ckpt_hash(path: str | None) -> str:
    import hashlib

    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha1()
    h.update(os.path.abspath(path).encode("utf-8"))
    st = os.stat(path)
    h.update(f"|{st.st_size}|{st.st_mtime_ns}".encode("utf-8"))
    return h.hexdigest()[:12]


def build_lookup_only_checkpoint(out_path: Path) -> Path:
    """Build a retokenizer checkpoint whose embedding table is set by the
    FVT lookup table and whose position embedding mirrors FG-CLIP 2's.
    No training is performed — this is the ALM-free 0.56 leaderboard hack
    equivalent expressed via the retokenizer wrapper.
    """
    if out_path.exists():
        print(f"[lookup-init] reusing {out_path}")
        return out_path

    from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTokenizer

    from self_training.retokenizer_model import FgClip2Retokenizer
    from validate import _patch_fgclip2_text_embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[lookup-init] loading FG-CLIP 2 base ...")
    base = AutoModelForCausalLM.from_pretrained(
        "qihoo360/fg-clip2-base", trust_remote_code=True
    )
    _patch_fgclip2_text_embeddings(base)
    base = base.to(device).eval()

    gemma_tok = AutoTokenizer.from_pretrained(
        "qihoo360/fg-clip2-base", trust_remote_code=True
    )
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    student = FgClip2Retokenizer(base, with_adapter=False).to(device).eval()
    student.init_from_lookup_table(
        clip_tokenizer=clip_tok,
        gemma_tokenizer=gemma_tok,
        gemma_token_embedding=base.text_model.embeddings.token_embedding,
        verbose=True,
    )
    with torch.no_grad():
        src = base.text_model.embeddings.position_embedding.weight.detach()
        student.position_embedding.weight.copy_(src.to(student.position_embedding.weight.device))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_state_dict": student.trainable_state_dict(),
            "lookup_only": True,
            "calibration": True,
        },
        out_path,
    )
    print(f"[lookup-init] saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
def run_eval(model_key: str, retokenizer_ckpt: str | None = None,
             dataset: str = "mscoco", batch_size: int = 32,
             cache_dir: str | None = None) -> dict:
    """Run validate.py's MS-COCO retrieval eval and return the metrics dict."""
    import hashlib

    from lpcvc_contract import CONTRACT_VERSION

    device = V.auto_device()
    spec = MODELS[model_key]
    if retokenizer_ckpt is not None:
        from dataclasses import replace as dc_replace

        spec = dc_replace(spec, retokenizer_checkpoint=retokenizer_ckpt)
        MODELS[model_key] = spec

    print(f"\n=== {model_key} on {dataset} ===")
    t0 = time.perf_counter()
    model, preprocess, tokenizer = V.load_model(model_key, device)
    print(f"   loaded in {time.perf_counter() - t0:.1f}s")

    # Build cache signatures (mirror validate.py main()).
    preprocess_sig = hashlib.sha1(
        f"{repr(preprocess)}|{CONTRACT_VERSION}".encode("utf-8")
    ).hexdigest()[:12]
    if spec.loader == "fgclip2_retokenized_hf":
        ckpt_h = _ckpt_hash(spec.retokenizer_checkpoint)
        token_sig = _hashlib_token_sig(model_key, spec, ckpt_h)
        model_sig = f"{model_key}|fgclip2_retokenized_short64_v1|{ckpt_h}"
    elif spec.loader == "fgclip2_hf":
        token_sig = _hashlib_token_sig(model_key, spec, "")
        model_sig = f"{model_key}|fgclip2_long196_nativeres_v4"
    else:
        token_sig = _hashlib_token_sig(model_key, spec, "")
        model_sig = model_key

    cache = V.CacheManager(
        root_dir=cache_dir or str(_REPO_ROOT / ".cache" / "validate"),
        enabled=True,
        model_sig=model_sig,
        preprocess_sig=preprocess_sig,
        token_sig=token_sig,
    )

    print(f"   loading {dataset} dataset ...")
    if dataset == "mscoco":
        payload = V.load_mscoco(str(_REPO_ROOT / "hf_cache"))
    else:
        raise ValueError(f"unsupported dataset {dataset}")

    print(f"   {len(payload[0])} images, {len(payload[1])} captions")
    scores = V.eval_retrieval(model, preprocess, tokenizer, payload, device, batch_size, cache)
    print(f"   scores: { {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()} }")
    return scores


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-model", default="mobileclip2_s2")
    p.add_argument("--lookup-ckpt-out",
                   default=str(_REPO_ROOT / "checkpoints" / "retokenizer" / "lookup_only.pt"))
    p.add_argument("--out", default=str(_REPO_ROOT / ".cache" / "calibration.json"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-lookup", action="store_true")
    args = p.parse_args()

    results: dict = {}

    # 1. Baseline.
    if not args.skip_baseline:
        scores = run_eval(args.baseline_model, dataset="mscoco", batch_size=args.batch_size)
        results["baseline_COCO_R10"] = float(scores.get("R@10", -1))
        results["baseline_R1"] = float(scores.get("R@1", -1))
        results["baseline_R5"] = float(scores.get("R@5", -1))

    # 2. Lookup hack — build the lookup-only checkpoint, then run.
    if not args.skip_lookup:
        ckpt_path = build_lookup_only_checkpoint(Path(args.lookup_ckpt_out))
        scores_lh = run_eval("fgclip2_base_retokenized",
                             retokenizer_ckpt=str(ckpt_path),
                             dataset="mscoco", batch_size=args.batch_size)
        results["lookup_hack_COCO_R10"] = float(scores_lh.get("R@10", -1))
        results["lookup_hack_R1"] = float(scores_lh.get("R@1", -1))
        results["lookup_hack_R5"] = float(scores_lh.get("R@5", -1))

    if "baseline_COCO_R10" in results and "lookup_hack_COCO_R10" in results:
        kill_at = max(results["lookup_hack_COCO_R10"], results["baseline_COCO_R10"] - 0.05)
        results["kill_at"] = kill_at
    results["calibrated_at"] = datetime.utcnow().isoformat() + "Z"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[calibrate] wrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
