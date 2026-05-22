from __future__ import annotations

import argparse
from pathlib import Path

from finger_pressure.config import Config
from finger_pressure.pipeline import run_pipeline


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="멀티모달 손가락 압력 회귀 학습")
    parser.add_argument("--data-root", type=Path, default=Path("data/04.수확 데이터"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    parser.add_argument("--no-amp", action="store_true", help="CUDA mixed precision 학습을 비활성화합니다.")
    args = parser.parse_args()

    return Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        baseline_lr=args.baseline_lr,
        use_amp=not args.no_amp,
    )


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
