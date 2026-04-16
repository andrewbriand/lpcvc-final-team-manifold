import io
import json
import tarfile
import tempfile
from pathlib import Path

from self_training.data_pipeline import extract_captions_from_shards, create_image_dataset


def _make_mock_shard(path: Path, n: int = 5):
    """Create a minimal webdataset tar mimicking pixparse/cc12m-wds format."""
    with tarfile.open(path, "w") as tar:
        for i in range(n):
            key = f"{i:09d}"
            from PIL import Image
            img_buf = io.BytesIO()
            Image.new("RGB", (4, 4), color=(i, i, i)).save(img_buf, format="JPEG")
            img_bytes = img_buf.getvalue()
            info = tarfile.TarInfo(name=f"{key}.jpg")
            info.size = len(img_bytes)
            tar.addfile(info, io.BytesIO(img_bytes))
            meta = {"caption": f"a photo of item {i}", "url": f"http://example.com/{i}"}
            meta_bytes = json.dumps(meta).encode()
            info = tarfile.TarInfo(name=f"{key}.json")
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))


def test_extract_captions():
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "shard-0000.tar"
        _make_mock_shard(shard_path, n=5)
        captions, source_keys = extract_captions_from_shards([shard_path])
        assert len(captions) == 5
        assert len(source_keys) == 5
        assert captions[0] == "a photo of item 0"
        assert source_keys[0] != source_keys[1]


def test_create_image_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "shard-0000.tar"
        _make_mock_shard(shard_path, n=3)
        ds = create_image_dataset([shard_path], shuffle=0)
        sample = next(iter(ds))
        assert len(sample) == 2  # (image, meta)
