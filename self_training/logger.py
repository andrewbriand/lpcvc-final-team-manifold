"""Structured training logger for the self-training pipeline.

Writes:
  - training_log.jsonl  — one JSON object per event (step / eval / checkpoint)
  - training_metrics.csv — summary row per eval event
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


_CSV_COLUMNS = ["step", "epoch", "loss", "lr", "coco_r10", "flickr_r10", "sample_r10", "timestamp"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingLogger:
    """Structured logger that writes JSONL events and a CSV metrics summary."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self.log_dir / "training_log.jsonl"
        self._csv_path = self.log_dir / "training_metrics.csv"

        self._jsonl_file = self._jsonl_path.open("a", encoding="utf-8")

        # Open CSV and write header only if the file is new / empty.
        csv_is_new = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        self._csv_file = self._csv_path.open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_COLUMNS)
        if csv_is_new:
            self._csv_writer.writeheader()

        # Cache the most-recent step-level loss/lr so we can include them in
        # eval rows without requiring callers to pass them again.
        self._last_loss: float | None = None
        self._last_lr: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(self, step: int, epoch: int, loss: float, lr: float, batch_size: int) -> None:
        """Write a training-step event to JSONL and print every 50 steps."""
        self._last_loss = loss
        self._last_lr = lr

        record = {
            "event": "step",
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
            "batch_size": batch_size,
            "timestamp": _now(),
        }
        self._write_jsonl(record)

        if step % 50 == 0:
            print(
                f"[step {step:>6d} | epoch {epoch}] loss={loss:.4f}  lr={lr:.2e}  bs={batch_size}"
            )

    def log_eval(self, step: int, epoch: int, eval_results: dict) -> None:
        """Write an eval event to JSONL and a summary row to CSV."""
        ts = _now()

        record = {
            "event": "eval",
            "step": step,
            "epoch": epoch,
            "datasets": {ds: metrics for ds, metrics in eval_results.items()},
            "timestamp": ts,
        }
        self._write_jsonl(record)

        # Print a concise summary.
        print(f"[eval  step={step} epoch={epoch}]")
        for dataset, metrics in eval_results.items():
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {dataset}: {metrics_str}")

        # Write one CSV summary row with the most useful R@10 numbers.
        csv_row = {
            "step": step,
            "epoch": epoch,
            "loss": self._last_loss if self._last_loss is not None else "",
            "lr": self._last_lr if self._last_lr is not None else "",
            "coco_r10": self._extract_r10(eval_results, "mscoco"),
            "flickr_r10": self._extract_r10(eval_results, "flickr30k"),
            "sample_r10": self._extract_r10(eval_results, "sample"),
            "timestamp": ts,
        }
        self._csv_writer.writerow(csv_row)
        self._csv_file.flush()

    def log_checkpoint(self, step: int, epoch: int, path: str, metric: float) -> None:
        """Write a checkpoint event to JSONL."""
        record = {
            "event": "checkpoint",
            "step": step,
            "epoch": epoch,
            "path": str(path),
            "metric": metric,
            "timestamp": _now(),
        }
        self._write_jsonl(record)

    def close(self) -> None:
        """Flush and close all open file handles."""
        self._jsonl_file.flush()
        self._jsonl_file.close()
        self._csv_file.flush()
        self._csv_file.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_jsonl(self, record: dict) -> None:
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()

    @staticmethod
    def _extract_r10(eval_results: dict, dataset_key: str) -> str:
        """Return R@10 for a dataset if present, else empty string."""
        metrics = eval_results.get(dataset_key, {})
        for key in ("R@10", "r10", "R10"):
            if key in metrics:
                return metrics[key]
        return ""
