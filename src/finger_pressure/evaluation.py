from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from finger_pressure.config import Config
from finger_pressure.trainer import forward_model, move_batch_to_device


FINGER_COLUMNS = [f"finger_{idx}" for idx in range(1, 6)]


def evaluate_model(
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    테스트 세트에서 5개 손가락 전체에 대한 회귀 성능을 계산한다.

    - RMSE/MAE: 모든 프레임과 모든 손가락 값을 펼쳐서 전체 평균 오차를 계산한다.
    - finger별 RMSE/MAE: 각 손가락 출력 차원마다 별도 성능을 계산한다.
    - Perfect Match Accuracy: 한 프레임의 5개 손가락 오차가 모두 ±tolerance 이내면 성공으로 본다.
    """

    model.eval()
    target_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            device_batch = move_batch_to_device(batch, device)
            output = forward_model(model, device_batch, model_type).detach().cpu().numpy()
            target = batch["target"].numpy()

            target_batches.append(target)
            prediction_batches.append(output)

            abs_error = np.abs(output - target)
            frame_mse = np.mean((output - target) ** 2, axis=1)
            perfect_match = np.all(abs_error <= config.tolerance, axis=1)

            for idx in range(output.shape[0]):
                row: dict[str, Any] = {
                    "fruit_type": batch["fruit_type"][idx],
                    "image_file": batch["image_file"][idx],
                    "csv_path": batch["csv_path"][idx],
                    "frame_mse": float(frame_mse[idx]),
                    "frame_rmse": float(np.sqrt(frame_mse[idx])),
                    "frame_mae": float(np.mean(abs_error[idx])),
                    "perfect_match": bool(perfect_match[idx]),
                }
                for finger_idx, finger_name in enumerate(FINGER_COLUMNS):
                    row[f"target_{finger_name}"] = float(target[idx, finger_idx])
                    row[f"prediction_{finger_name}"] = float(output[idx, finger_idx])
                    row[f"residual_{finger_name}"] = float(output[idx, finger_idx] - target[idx, finger_idx])
                    row[f"abs_error_{finger_name}"] = float(abs_error[idx, finger_idx])
                rows.append(row)

    targets = np.concatenate(target_batches, axis=0)
    predictions = np.concatenate(prediction_batches, axis=0)
    errors = predictions - targets
    abs_errors = np.abs(errors)

    metrics: dict[str, float] = {
        "RMSE_all_outputs": float(np.sqrt(np.mean(errors**2))),
        "MAE_all_outputs": float(np.mean(abs_errors)),
        f"Perfect_Match_Accuracy_pm_{config.tolerance}": float(
            np.mean(np.all(abs_errors <= config.tolerance, axis=1))
        ),
    }

    for finger_idx, finger_name in enumerate(FINGER_COLUMNS):
        finger_errors = errors[:, finger_idx]
        metrics[f"RMSE_{finger_name}"] = float(np.sqrt(np.mean(finger_errors**2)))
        metrics[f"MAE_{finger_name}"] = float(np.mean(np.abs(finger_errors)))

    return metrics, pd.DataFrame(rows)


def print_failure_cases(prediction_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """5개 손가락 평균 MSE가 가장 큰 상위 프레임을 출력한다."""

    failure_df = prediction_df.sort_values("frame_mse", ascending=False).head(top_k)

    print("\n[Failure Analysis] 5개 손가락 평균 MSE가 가장 큰 상위 5개 프레임")
    for rank, (_, row) in enumerate(failure_df.iterrows(), start=1):
        targets = [row[f"target_{finger}"] for finger in FINGER_COLUMNS]
        predictions = [row[f"prediction_{finger}"] for finger in FINGER_COLUMNS]
        abs_errors = [row[f"abs_error_{finger}"] for finger in FINGER_COLUMNS]
        print(
            f"{rank}. fruit={row['fruit_type']} image={row['image_file']} "
            f"frame_mse={row['frame_mse']:.4f} frame_mae={row['frame_mae']:.4f}\n"
            f"   target={np.round(targets, 3).tolist()}\n"
            f"   pred  ={np.round(predictions, 3).tolist()}\n"
            f"   abs_e ={np.round(abs_errors, 3).tolist()}"
        )

    return failure_df
