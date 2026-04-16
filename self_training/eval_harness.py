from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from validate import (
    DATASETS,
    GROUPS,
    eval_binary,
    eval_itt,
    eval_retrieval,
)
from lpcvc_models import MODELS


def run_eval_suite(
    model,
    preprocess,
    tokenizer,
    datasets: list[str],
    device: str = "cuda",
    batch_size: int = 64,
    hf_cache_dir: str = "hf_cache",
    img_csv: str = "dataset/img_list.csv",
    txt_csv: str = "dataset/txt_list.csv",
    img_dir: str = "dataset/images",
) -> dict[str, dict[str, float]]:
    """Run evaluation on multiple datasets. Returns {dataset_key: {metric: value}}."""
    results = {}

    for ds_key in datasets:
        if ds_key not in DATASETS:
            if ds_key in GROUPS:
                for sub_key in GROUPS[ds_key]:
                    sub = _eval_single(
                        model, preprocess, tokenizer, sub_key, device,
                        batch_size, hf_cache_dir, img_csv, txt_csv, img_dir,
                    )
                    results[sub_key] = sub
                continue
            else:
                raise ValueError(f"Unknown dataset: {ds_key}")
        results[ds_key] = _eval_single(
            model, preprocess, tokenizer, ds_key, device,
            batch_size, hf_cache_dir, img_csv, txt_csv, img_dir,
        )

    return results


def _eval_single(
    model, preprocess, tokenizer, ds_key, device,
    batch_size, hf_cache_dir, img_csv, txt_csv, img_dir,
) -> dict[str, float]:
    task_type, source, _ = DATASETS[ds_key]
    cache = None

    # Import load functions here to avoid circular imports at module level
    from validate import load_sample, load_mscoco, load_flickr30k, load_sugarcrepe, load_sugarcrepe_pp

    if ds_key == "sample":
        payload = load_sample(img_csv, txt_csv, img_dir)
    elif ds_key == "mscoco":
        payload = load_mscoco(hf_cache_dir)
    elif ds_key == "flickr30k":
        payload = load_flickr30k(hf_cache_dir)
    elif task_type == "binary":
        payload = load_sugarcrepe(hf_cache_dir)
    elif task_type == "itt":
        # ds_key is e.g. "sugarcrepe_pp_swap_att" — strip the common prefix
        subset = ds_key[len("sugarcrepe_pp_"):]
        payload = load_sugarcrepe_pp(hf_cache_dir, subset)
    else:
        raise ValueError(f"Unknown dataset key or task type: {ds_key!r} / {task_type!r}")

    if task_type == "retrieval":
        return eval_retrieval(model, preprocess, tokenizer, payload, device, batch_size, cache)
    elif task_type == "binary":
        return eval_binary(model, preprocess, tokenizer, payload, device, batch_size, cache)
    elif task_type == "itt":
        return eval_itt(model, preprocess, tokenizer, payload, device, batch_size, cache)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
