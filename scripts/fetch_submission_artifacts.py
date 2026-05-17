#!/usr/bin/env python3
"""Download Team Manifold's final LPCVC Track 1 artifacts from Hugging Face.

The public Git repo intentionally does not commit model checkpoints, LoRA
adapter weights, or generated QAI Hub manifests. This script restores the
small set needed to reproduce the documented submission-family validation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO = "jrauvola/lpcvc2026-track1-team-manifold-final"

ARTIFACTS = (
    (
        "final_submission/retokenizer/best.pt",
        "retokenizer/best.pt",
    ),
    (
        "final_submission/stage1_best/adapter_config.json",
        "stage1_best/adapter_config.json",
    ),
    (
        "final_submission/stage1_best/adapter_model.safetensors",
        "stage1_best/adapter_model.safetensors",
    ),
    (
        "final_submission/stage1_best/README.md",
        "stage1_best/README.md",
    ),
    (
        "final_submission/fgclip2_schall_stage1/compile_manifest.json",
        "fgclip2_schall_stage1/compile_manifest.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out-dir", default="artifacts/submission")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for remote_path, local_rel in ARTIFACTS:
        cached = hf_hub_download(
            repo_id=args.repo_id,
            filename=remote_path,
            repo_type=args.repo_type,
            revision=args.revision,
        )
        target = out_dir / local_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, target)
        print(f"{remote_path} -> {target}")

    print("\nValidation command:")
    print(
        "python3 validate.py "
        "--model fgclip2_base_retokenized "
        f"--retokenizer-checkpoint {out_dir / 'retokenizer/best.pt'} "
        f"--schall-stage1-adapter {out_dir / 'stage1_best'} "
        "--fgclip2-fix-resolution "
        "--datasets sample"
    )


if __name__ == "__main__":
    main()
