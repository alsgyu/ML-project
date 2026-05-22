from __future__ import annotations

import joblib
import pandas as pd
import torch

from finger_pressure.config import Config
from finger_pressure.data import build_dataloaders, discover_raw_records, fit_tabular_features, make_splits
from finger_pressure.evaluation import evaluate_model, print_failure_cases
from finger_pressure.models import LateFusionResNetRegressor, TabularOnlyMLP, build_multimodal_optimizer
from finger_pressure.trainer import train_one_model
from finger_pressure.utils import set_seed


def run_pipeline(config: Config) -> None:
    """전체 데이터 준비, 학습, 평가, 실패 분석을 순서대로 실행한다."""

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[환경] device={device}")

    raw_sensor_df, metadata, fruit_types = discover_raw_records(config)
    print(f"[데이터] 필터링 후 샘플 수: {len(metadata):,}")
    print(f"[데이터] 과일 종류: {fruit_types}")
    print(f"[데이터] 센서 후보 컬럼 수: {raw_sensor_df.shape[1]:,}")

    splits = make_splits(metadata, config.seed)
    print("[데이터] split 크기:", {name: int(len(index)) for name, index in splits.items()})

    tabular_features, preprocessors = fit_tabular_features(raw_sensor_df, metadata, fruit_types, splits["train"], config)
    joblib.dump(preprocessors, config.output_dir / "tabular_preprocessors.joblib")

    loaders = build_dataloaders(metadata, tabular_features, splits, config)
    tabular_dim = tabular_features.shape[1]

    baseline_model = TabularOnlyMLP(input_dim=tabular_dim, output_dim=config.target_dim).to(device)
    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=config.baseline_lr,
        weight_decay=config.weight_decay,
    )
    baseline_history = train_one_model(
        baseline_model,
        "tabular",
        loaders,
        baseline_optimizer,
        config,
        config.output_dir / "best_tabular_mlp.pt",
        device,
    )
    baseline_history.to_csv(config.output_dir / "history_tabular_mlp.csv", index=False)

    multimodal_model = LateFusionResNetRegressor(
        tabular_dim=tabular_dim,
        output_dim=config.target_dim,
        pretrained=True,
    ).to(device)
    multimodal_optimizer = build_multimodal_optimizer(
        multimodal_model,
        backbone_lr=config.backbone_lr,
        head_lr=config.head_lr,
        weight_decay=config.weight_decay,
    )
    multimodal_history = train_one_model(
        multimodal_model,
        "multimodal",
        loaders,
        multimodal_optimizer,
        config,
        config.output_dir / "best_late_fusion_resnet18.pt",
        device,
    )
    multimodal_history.to_csv(config.output_dir / "history_late_fusion_resnet18.csv", index=False)

    baseline_metrics, baseline_predictions = evaluate_model(baseline_model, "tabular", loaders["test"], config, device)
    multimodal_metrics, multimodal_predictions = evaluate_model(
        multimodal_model, "multimodal", loaders["test"], config, device
    )

    comparison_df = pd.DataFrame(
        [
            {"model": "Tabular Only MLP", **baseline_metrics},
            {"model": "Late Fusion ResNet-18", **multimodal_metrics},
        ]
    )
    print("\n[테스트 성능 비교]")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(config.output_dir / "model_comparison_metrics.csv", index=False)
    baseline_predictions.to_csv(config.output_dir / "test_predictions_tabular_mlp.csv", index=False)
    multimodal_predictions.to_csv(config.output_dir / "test_predictions_late_fusion_resnet18.csv", index=False)

    failure_df = print_failure_cases(multimodal_predictions, top_k=5)
    failure_df.to_csv(config.output_dir / "top5_failure_cases.csv", index=False)

    print(f"\n[완료] 결과 파일 저장 위치: {config.output_dir.resolve()}")
