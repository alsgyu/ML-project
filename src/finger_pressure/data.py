from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from finger_pressure.config import Config
from finger_pressure.utils import parse_finger_pressure


NON_SENSOR_COLUMNS = {"finger_pressure", "Time", "Timestamp"}


def get_fruit_type(csv_path: Path, data_root: Path) -> str:
    """CSV 경로에서 data_root 바로 아래의 과일 폴더명을 추출한다."""

    relative_parts = csv_path.relative_to(data_root).parts
    if len(relative_parts) < 3:
        raise ValueError(f"예상과 다른 폴더 구조입니다: {csv_path}")
    return relative_parts[0]


def rgb_image_path_for_csv(csv_path: Path) -> Path:
    """CSV와 같은 stem을 갖는 RGB 카메라 이미지(_R.jpeg)를 반환한다."""

    return csv_path.with_name(f"{csv_path.stem}_R.jpeg")


def discover_raw_records(config: Config) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    """os.walk로 CSV/RGB JPEG 쌍을 탐색하고 압력 필터를 통과한 샘플만 수집한다."""

    if not config.data_root.exists():
        raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다: {config.data_root}")

    fruit_types = sorted(path.name for path in config.data_root.iterdir() if path.is_dir())
    if len(fruit_types) != 5:
        raise ValueError(f"과일 원핫 인코딩은 5차원을 기대합니다. 현재 과일 폴더 수: {len(fruit_types)}")

    sensor_rows: list[pd.Series] = []
    metadata: list[dict[str, Any]] = []

    for current_dir, _, filenames in os.walk(config.data_root):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue

            csv_path = Path(current_dir) / filename
            image_path = rgb_image_path_for_csv(csv_path)
            if not image_path.exists():
                continue

            try:
                row = pd.read_csv(csv_path, nrows=1, encoding="utf-8-sig").iloc[0]
            except Exception as exc:
                print(f"[경고] CSV 읽기 실패: {csv_path} ({exc})")
                continue

            pressure_values = parse_finger_pressure(row.get("finger_pressure"))
            if pressure_values is None:
                continue
            if float(np.mean(pressure_values)) < config.pressure_mean_threshold:
                continue

            raw_sensor = row.drop(labels=list(NON_SENSOR_COLUMNS), errors="ignore")
            numeric_sensor = pd.to_numeric(raw_sensor, errors="coerce")
            fruit_type = get_fruit_type(csv_path, config.data_root)

            sensor_rows.append(numeric_sensor)
            metadata.append(
                {
                    "csv_path": str(csv_path),
                    "image_path": str(image_path),
                    "image_file": image_path.name,
                    "fruit_type": fruit_type,
                    "target_pressures": [float(value) for value in pressure_values],
                    "pressure_mean": float(np.mean(pressure_values)),
                }
            )

    if not metadata:
        raise RuntimeError("유효한 CSV/JPEG 샘플을 찾지 못했습니다.")

    raw_sensor_df = pd.DataFrame(sensor_rows).replace([np.inf, -np.inf], np.nan)
    return raw_sensor_df, metadata, fruit_types


def make_splits(metadata: list[dict[str, Any]], seed: int) -> dict[str, np.ndarray]:
    """전체 샘플 인덱스를 Train 60%, Val 20%, Test 20%로 나눈다."""

    indices = np.arange(len(metadata))
    fruit_labels = np.array([item["fruit_type"] for item in metadata])

    try:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.4, random_state=seed, shuffle=True, stratify=fruit_labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=seed,
            shuffle=True,
            stratify=fruit_labels[temp_idx],
        )
    except ValueError as exc:
        print(f"[경고] stratified split 실패, random split으로 대체합니다: {exc}")
        train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=seed, shuffle=True)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, shuffle=True)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def fit_tabular_features(
    raw_sensor_df: pd.DataFrame,
    metadata: list[dict[str, Any]],
    fruit_types: list[str],
    train_idx: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train split에만 imputer/scaler/PCA를 fit하고 전체 샘플의 20차원 tabular feature를 만든다."""

    if len(train_idx) < config.pca_dim:
        raise ValueError(f"PCA {config.pca_dim}차원을 학습하려면 train 샘플이 최소 {config.pca_dim}개 필요합니다.")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=config.pca_dim, random_state=config.seed)

    train_imputed = imputer.fit_transform(raw_sensor_df.iloc[train_idx])
    train_scaled = scaler.fit_transform(train_imputed)
    pca.fit(train_scaled)

    sensor_pca = pca.transform(scaler.transform(imputer.transform(raw_sensor_df))).astype(np.float32)

    fruit_to_idx = {fruit: idx for idx, fruit in enumerate(fruit_types)}
    fruit_onehot = np.zeros((len(metadata), len(fruit_types)), dtype=np.float32)
    for row_idx, item in enumerate(metadata):
        fruit_onehot[row_idx, fruit_to_idx[item["fruit_type"]]] = 1.0

    tabular_features = np.concatenate([sensor_pca, fruit_onehot], axis=1)
    preprocessors = {
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "fruit_types": fruit_types,
        "fruit_to_idx": fruit_to_idx,
        "sensor_columns": raw_sensor_df.columns.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return tabular_features, preprocessors


class MultimodalFruitDataset(Dataset):
    """이미지, 정형 데이터, 회귀 타겟, 메타데이터를 함께 반환하는 Custom Dataset."""

    def __init__(
        self,
        indices: np.ndarray,
        metadata: list[dict[str, Any]],
        tabular_features: np.ndarray,
        image_transform: transforms.Compose,
    ) -> None:
        self.indices = indices
        self.metadata = metadata
        self.tabular_features = tabular_features
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, dataset_index: int) -> dict[str, Any]:
        sample_index = int(self.indices[dataset_index])
        item = self.metadata[sample_index]

        image = Image.open(item["image_path"]).convert("RGB")
        return {
            "image": self.image_transform(image),
            "tabular": torch.tensor(self.tabular_features[sample_index], dtype=torch.float32),
            # 5개 손가락 압력을 동시에 예측하는 multi-output regression 타겟이다.
            "target": torch.tensor(item["target_pressures"], dtype=torch.float32),
            "fruit_type": item["fruit_type"],
            "image_file": item["image_file"],
            "csv_path": item["csv_path"],
        }


def build_transforms(config: Config) -> dict[str, transforms.Compose]:
    """ResNet-18 사전학습 가중치와 호환되는 이미지 전처리를 정의한다."""

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    common = [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
    return {"train": transforms.Compose(common), "eval": transforms.Compose(common)}


def build_dataloaders(
    metadata: list[dict[str, Any]],
    tabular_features: np.ndarray,
    splits: dict[str, np.ndarray],
    config: Config,
) -> dict[str, DataLoader]:
    """Train/Val/Test Dataset과 DataLoader를 생성한다."""

    image_transforms = build_transforms(config)
    datasets = {
        "train": MultimodalFruitDataset(splits["train"], metadata, tabular_features, image_transforms["train"]),
        "val": MultimodalFruitDataset(splits["val"], metadata, tabular_features, image_transforms["eval"]),
        "test": MultimodalFruitDataset(splits["test"], metadata, tabular_features, image_transforms["eval"]),
    }
    return {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }
