"""Lightweight loguru + TensorBoard integration, one instance per training run."""
from pathlib import Path
from loguru import logger


class RunLogger:
    def __init__(self, name: str, enabled: bool = True):
        self.writer = None
        self.enabled = enabled
        if enabled:
            from torch.utils.tensorboard import SummaryWriter
            path = Path("runs") / name
            path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(path))
            logger.info("tensorboard --logdir runs | experiment: {}", name)

    def scalar(self, tag: str, value, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)
        if step == 0 or (step + 1) % 20 == 0:
            logger.info("{} step={} value={:.4f}", tag, step, float(value))

    def close(self):
        if self.writer is not None:
            self.writer.close()
