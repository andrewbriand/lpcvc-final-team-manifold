"""Push training artifacts (LoRA adapter, ONNX, manifests, logs) to a private HF Hub repo.

Designed to run on a GPU box with good network — Lambda's wifi is way faster
than uploading from a Mac, and we want artifacts persisted before tearing down
the instance.

Usage:
    python scripts/push_to_hf.py <repo_id> <src_dir> [<src_dir> ...]

Reads HF_TOKEN from ~/lpcvc/.env or environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_paths = [Path("/home/ubuntu/lpcvc/.env"), Path(".env"), Path(__file__).resolve().parent.parent / ".env"]
    for p in env_paths:
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "HF_TOKEN":
                    return v.strip().strip('"').strip("'")
    raise RuntimeError("HF_TOKEN not found in env or .env")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="HF repo id, e.g. jrauvola/lpcvc2026-track1-schall-v2")
    parser.add_argument("src_dirs", nargs="+", help="One or more local dirs to upload")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_true", help="Override --private")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    args = parser.parse_args()

    token = _load_token()
    private = args.private and not args.public

    from huggingface_hub import HfApi, create_repo, upload_folder

    api = HfApi(token=token)

    # Resolve user namespace if repo_id is bare
    if "/" not in args.repo_id:
        try:
            who = api.whoami()
            user = who.get("name") or who.get("id")
            if not user:
                raise RuntimeError("could not resolve HF username; pass username/repo_id")
            repo_id = f"{user}/{args.repo_id}"
        except Exception as e:
            print(f"[push] whoami failed: {e}; using repo_id as-is")
            repo_id = args.repo_id
    else:
        repo_id = args.repo_id

    print(f"[push] repo_id={repo_id} private={private} type={args.repo_type}")
    create_repo(
        repo_id, repo_type=args.repo_type, private=private, exist_ok=True, token=token
    )

    for src in args.src_dirs:
        src_path = Path(src)
        if not src_path.exists():
            print(f"[push] SKIP missing dir: {src}")
            continue
        path_in_repo = src_path.name
        print(f"[push] uploading {src_path} -> {repo_id}/{path_in_repo}/")
        upload_folder(
            folder_path=str(src_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=args.repo_type,
            token=token,
            commit_message=f"push artifacts: {path_in_repo}",
            ignore_patterns=["__pycache__/*", "*.pyc"],
        )
        print(f"[push] done: {path_in_repo}")

    print(f"[push] ALL DONE. Browse at https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
