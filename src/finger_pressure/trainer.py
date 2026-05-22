from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from finger_pressure.config import Config


class EarlyStopping:
    """Validation loss가 개선되지 않으면 학습을 조기 종료하고 최적 가중치를 저장한다."""

    def __init__(self, patience: int, checkpoint_path: Path) -> None:
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_loss = math.inf
        self.counter = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return False

        self.counter += 1
        return self.counter >= self.patience


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """텐서 항목만 GPU/CPU 장치로 이동한다."""

    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def forward_model(model: nn.Module, batch: dict[str, Any], model_type: str) -> torch.Tensor:
    """모델 종류에 따라 필요한 입력만 전달한다."""

    if model_type == "tabular":
        return model(batch["tabular"])
    if model_type == "multimodal":
        return model(batch["image"], batch["tabular"])
    raise ValueError(f"알 수 없는 model_type: {model_type}")


def train_one_model(
    model: nn.Module,
    model_type: str,
    loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    config: Config,
    checkpoint_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    """Train/Validation loop와 EarlyStopping을 수행한다."""

    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(config.patience, checkpoint_path)
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch in loaders["train"]:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            # CUDA 환경에서는 mixed precision을 사용해 GPU 메모리 사용량을 줄이고 처리량을 높인다.
            # CPU 환경에서는 자동으로 비활성화되어 동일한 코드가 안전하게 동작한다.
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                predictions = forward_model(model, batch, model_type)
                loss = criterion(predictions, batch["target"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = move_batch_to_device(batch, device)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    predictions = forward_model(model, batch, model_type)
                    loss = criterion(predictions, batch["target"])
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        print(f"[{model_type}] Epoch {epoch:03d}/{config.epochs} train_mse={train_loss:.5f} val_mse={val_loss:.5f}")

        if early_stopping.step(val_loss, model):
            print(f"[{model_type}] EarlyStopping: {config.patience} epochs 동안 개선 없음")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return pd.DataFrame(history)
