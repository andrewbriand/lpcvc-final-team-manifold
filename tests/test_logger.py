import tempfile
from pathlib import Path
from self_training.logger import TrainingLogger


def test_logger_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = TrainingLogger(log_dir)
        logger.log_step(step=1, epoch=0, loss=2.5, lr=5e-4, batch_size=8192)
        logger.log_eval(step=100, epoch=1, eval_results={
            "mscoco": {"R@1": 0.3, "R@10": 0.6},
            "flickr30k": {"R@10": 0.7},
        })
        logger.log_checkpoint(step=100, epoch=1, path="/tmp/ckpt.pt", metric=0.6)
        logger.close()
        assert (log_dir / "training_log.jsonl").exists()
        assert (log_dir / "training_metrics.csv").exists()
        import json
        lines = (log_dir / "training_log.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["event"] == "step"
