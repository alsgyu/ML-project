from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """실험 전체에서 공유하는 주요 하이퍼파라미터."""

    data_root: Path = Path("data/04.수확 데이터")
    output_dir: Path = Path("outputs")
    image_size: int = 224
    pca_dim: int = 15
    target_dim: int = 5
    batch_size: int = 16
    num_workers: int = 2
    epochs: int = 50
    patience: int = 8
    seed: int = 42
    pressure_mean_threshold: float = 0.5
    tolerance: float = 0.2
    baseline_lr: float = 1e-3
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
