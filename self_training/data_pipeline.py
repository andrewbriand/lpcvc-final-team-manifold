from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import webdataset as wds
from PIL import Image


def extract_captions_from_shards(
    shard_paths: list[Path],
) -> tuple[list[str], list[str]]:
    """Extract all captions and unique source keys from CC12M webdataset shards.

    Returns:
        captions: list of caption strings
        source_keys: list of unique keys (shard_idx:sample_key) for own-caption exclusion
    """
    captions: list[str] = []
    source_keys: list[str] = []

    for shard_idx, shard_path in enumerate(sorted(shard_paths)):
        with tarfile.open(shard_path, "r") as tar:
            json_members = [m for m in tar.getmembers() if m.name.endswith(".json")]
            for member in json_members:
                f = tar.extractfile(member)
                if f is None:
                    continue
                meta = json.loads(f.read())
                caption = meta.get("caption", "").strip()
                if not caption:
                    continue
                sample_key = member.name.rsplit(".", 1)[0]
                captions.append(caption)
                source_keys.append(f"{shard_idx}:{sample_key}")

    return captions, source_keys


def create_image_dataset(
    shard_paths: list[Path],
    shuffle: int = 1000,
) -> wds.WebDataset:
    """Create a webdataset iterator over images + metadata from CC12M shards."""
    urls = [str(p) for p in sorted(shard_paths)]
    dataset = (
        wds.WebDataset(urls, shardshuffle=True)
        .shuffle(shuffle)
        .decode("pil")
        .to_tuple("jpg;png", "json")
        .map_tuple(lambda img: img.convert("RGB"), lambda meta: meta)
    )
    return dataset


def list_shard_files(shards_dir: Path, num_shards: int | None = None) -> list[Path]:
    """List .tar shard files in a directory, optionally limited to first N."""
    shards = sorted(shards_dir.glob("*.tar"))
    if num_shards is not None:
        shards = shards[:num_shards]
    return shards
