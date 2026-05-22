# Multi-output End-to-End Multimodal Finger Pressure Regression

과일 RGB 이미지와 로봇 센서 데이터를 함께 사용해 로봇 손가락 5개의 압력을 동시에 예측하는 PyTorch 기반 다중 출력 멀티모달 회귀 프로젝트입니다. 최종 타겟은 `finger_pressure` 컬럼에서 파싱한 5차원 벡터입니다.

```text
y = [finger_1, finger_2, finger_3, finger_4, finger_5]
```

## 1. 프로젝트 목표

로봇이 과일을 수확할 때 손가락별 압력은 과일 손상 가능성과 직접적으로 연결됩니다. 본 프로젝트는 수확 상황에서 기록된 카메라 이미지와 약 700개 규모의 센서 데이터를 결합하여 5개 손가락 압력을 한 번에 예측하는 multi-output end-to-end multimodal regression 파이프라인을 구현합니다.

비교 실험은 다음 두 모델로 구성됩니다.

| 모델 | 입력 데이터 | 출력 |
|---|---|---|
| Tabular Only MLP | 센서 PCA 15차원 + 과일 one-hot 5차원 | 5개 손가락 압력 |
| Late Fusion ResNet-18 | RGB 이미지 + 20차원 정형 데이터 | 5개 손가락 압력 |

## 2. 데이터 구조

원본 데이터는 아래 구조를 가정합니다.

```text
data/
└── 04.수확 데이터/
    ├── 01.사과/
    │   └── 251111_APPL_0889/
    │       ├── 251111_APPL_0889_001.csv
    │       ├── 251111_APPL_0889_001_R.jpeg
    │       └── 251111_APPL_0889_001_D.jpeg
    ├── 02.배(수확)/
    ├── 03.배(비수확)/
    ├── 04.복숭아/
    └── 05.자두/
```

CSV와 이미지는 같은 파일 stem을 기준으로 매칭합니다. 본 실험에서는 RGB 카메라 이미지인 `*_R.jpeg`만 사용하고, 깊이 이미지인 `*_D.jpeg`는 사용하지 않습니다.

## 3. 전처리

CSV의 0번째 행은 헤더이고 1번째 행이 실제 데이터입니다. `finger_pressure` 컬럼은 `"2.2, 1.8, 2.1, 0.8, 0.4"` 형태의 문자열이므로 정규식으로 숫자를 추출해 5개 float 값으로 변환합니다.

손을 뻗는 과정처럼 접촉이 약한 프레임을 제외하기 위해 5개 압력 평균이 `0.5` 이상인 프레임만 학습에 사용합니다.

정형 데이터 전처리는 다음 순서로 수행합니다.

```text
약 700개 센서 컬럼
-> median imputation
-> StandardScaler
-> PCA 15차원
-> 과일 종류 one-hot 5차원 결합
-> 최종 tabular feature 20차원
```

데이터 누수를 방지하기 위해 `SimpleImputer`, `StandardScaler`, `PCA`는 train split에만 fit하고 validation/test에는 transform만 적용합니다.

이미지는 ResNet-18 입력 규격에 맞춰 `224x224` 리사이즈, tensor 변환, ImageNet normalization을 적용합니다.

## 4. 모델 구조

### Tabular Only MLP

베이스라인 모델은 20차원 정형 데이터만 입력받아 5개 손가락 압력을 동시에 출력합니다.

```text
20-dim tabular input
-> Linear(20, 64) + LayerNorm + ReLU + Dropout
-> Linear(64, 32) + ReLU + Dropout
-> Linear(32, 16) + ReLU
-> Linear(16, 5)
```

### Late Fusion ResNet-18

메인 모델은 pretrained ResNet-18의 마지막 classification layer를 제거하여 512차원 이미지 특징을 추출합니다. 이후 20차원 tabular feature와 concat하여 총 532차원 fusion vector를 만들고, regression head가 5개 압력을 동시에 출력합니다.

```text
RGB image
-> Pretrained ResNet-18 without final FC
-> 512-dim image feature

20-dim tabular feature

[512 image feature ; 20 tabular feature]
-> 532-dim late fusion vector
-> Regression head
-> 5-dim pressure vector
```

Fine-tuning에서는 pretrained backbone과 새로 추가한 regression head에 차등 학습률을 적용합니다.

| 파라미터 그룹 | Learning rate |
|---|---:|
| ResNet-18 backbone | `1e-5` |
| Regression head | `1e-3` |

## 5. 평가 지표

테스트 세트에서 다음 지표를 계산합니다.

| 지표 | 설명 |
|---|---|
| `RMSE_all_outputs` | 모든 프레임과 5개 손가락 출력을 펼쳐 계산한 전체 RMSE |
| `MAE_all_outputs` | 모든 프레임과 5개 손가락 출력을 펼쳐 계산한 전체 MAE |
| `Perfect_Match_Accuracy_pm_0.2` | 한 프레임의 5개 손가락 예측 오차가 모두 ±0.2 이내일 때만 성공 |
| `RMSE_finger_i`, `MAE_finger_i` | 손가락별 RMSE/MAE |

Perfect Match Accuracy는 손가락별 오차를 따로 보는 기준보다 훨씬 엄격합니다. 예를 들어 5개 중 4개 손가락이 정확해도 하나가 ±0.2를 벗어나면 해당 프레임은 실패로 계산됩니다.

## 6. 실패 케이스 분석

테스트 세트에서 5개 손가락의 평균 MSE가 가장 큰 상위 5개 프레임을 출력합니다. 각 프레임에 대해 다음 정보를 저장합니다.

- 과일 종류
- 이미지 파일명
- 5개 실제 압력값
- 5개 예측 압력값
- 손가락별 절대 오차
- 프레임 평균 MSE/MAE

이 결과는 보고서의 Discussion에서 어떤 과일, 어떤 압력 패턴, 어떤 손가락에서 모델이 취약한지 분석하는 데 사용할 수 있습니다.

## 7. 저장소 구조

```text
.
├── README.md
├── requirements.txt
├── multimodal_finger_pressure.py      # 실행용 thin wrapper
├── src/
│   └── finger_pressure/
│       ├── __init__.py
│       ├── cli.py                     # CLI argument parsing
│       ├── config.py                  # 실험 설정 dataclass
│       ├── data.py                    # 데이터 탐색, 전처리, Dataset/DataLoader
│       ├── evaluation.py              # multi-output 평가와 failure analysis
│       ├── models.py                  # 5출력 MLP, 5출력 Late Fusion ResNet-18
│       ├── pipeline.py                # 전체 학습/평가 orchestration
│       ├── trainer.py                 # train/validation loop, EarlyStopping
│       └── utils.py                   # seed 고정, pressure parser
├── data/                              # 로컬 원본 데이터, Git commit 제외 권장
└── outputs/                           # 학습 결과, Git commit 제외 권장
```

## 8. 실행 방법

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 multimodal_finger_pressure.py --epochs 50 --batch-size 16
```

고성능 GPU 환경에서는 batch size를 늘려 학습 속도를 높일 수 있습니다.

```bash
python3 multimodal_finger_pressure.py --epochs 50 --batch-size 32 --num-workers 4
```

기본적으로 CUDA 환경에서는 mixed precision 학습이 활성화됩니다. GPU 메모리 여유가 충분하거나 수치 안정성 비교 실험이 필요하면 아래 옵션으로 비활성화할 수 있습니다.

```bash
python3 multimodal_finger_pressure.py --no-amp
```

## 9. 출력 파일

실행 후 `outputs/`에 다음 파일이 생성됩니다.

```text
outputs/
├── best_tabular_mlp.pt
├── best_late_fusion_resnet18.pt
├── history_tabular_mlp.csv
├── history_late_fusion_resnet18.csv
├── model_comparison_metrics.csv
├── test_predictions_tabular_mlp.csv
├── test_predictions_late_fusion_resnet18.csv
├── tabular_preprocessors.joblib
└── top5_failure_cases.csv
```

## 10. 실험 결과

본 실험에서는 평균 압력이 `0.5` 이상인 유효 프레임 810개를 사용했습니다. 데이터는 train 486개, validation 162개, test 162개로 분할했습니다. 평가는 테스트 세트에서 수행했으며, 5개 손가락 출력을 모두 포함한 multi-output 회귀 성능을 비교했습니다.

### 10.1 전체 성능 비교

| 모델 | RMSE 전체 | MAE 전체 | Perfect Match Accuracy |
|---|---:|---:|---:|
| Tabular Only MLP | 0.3771 | 0.2952 | 3.09% |
| Late Fusion ResNet-18 | 0.3706 | 0.2874 | 1.23% |

Late Fusion ResNet-18은 Tabular Only MLP 대비 전체 RMSE를 약 1.70%, 전체 MAE를 약 2.66% 개선했습니다. 즉 이미지 특징을 센서 기반 정형 특징과 결합했을 때 5개 손가락 압력의 평균적인 회귀 오차는 소폭 감소했습니다.

다만 Perfect Match Accuracy는 3.09%에서 1.23%로 감소했습니다. 이 지표는 한 프레임에서 5개 손가락이 모두 ±0.2 이내에 들어와야 성공으로 인정하는 매우 엄격한 기준입니다. 따라서 메인 모델은 평균적인 오차를 줄이는 데는 도움이 되었지만, 5개 손가락을 동시에 매우 정밀하게 맞히는 능력은 아직 충분하지 않았다고 해석할 수 있습니다.

### 10.2 손가락별 성능 비교

| 손가락 | Tabular RMSE | Multimodal RMSE | RMSE 변화 | Tabular MAE | Multimodal MAE | MAE 변화 |
|---|---:|---:|---:|---:|---:|---:|
| Finger 1 | 0.4917 | 0.4749 | 3.41% 개선 | 0.3959 | 0.3822 | 3.44% 개선 |
| Finger 2 | 0.4468 | 0.4158 | 6.93% 개선 | 0.3669 | 0.3355 | 8.57% 개선 |
| Finger 3 | 0.3989 | 0.4159 | 4.27% 악화 | 0.3266 | 0.3292 | 0.81% 악화 |
| Finger 4 | 0.2540 | 0.2717 | 6.97% 악화 | 0.2088 | 0.2217 | 6.14% 악화 |
| Finger 5 | 0.2143 | 0.2040 | 4.78% 개선 | 0.1781 | 0.1684 | 5.45% 개선 |

손가락별로 보면 이미지 정보를 추가한 효과가 균일하지 않았습니다. Finger 1, Finger 2, Finger 5에서는 멀티모달 모델이 베이스라인보다 낮은 오차를 보였고, 특히 Finger 2의 MAE는 약 8.57% 개선되었습니다. 이는 손가락 1, 2, 5의 압력 변화가 이미지에서 관찰되는 과일 위치, 접촉 상태, 수확 장면의 시각적 특징과 어느 정도 연관되어 있음을 시사합니다.

반면 Finger 3과 Finger 4에서는 메인 모델의 성능이 오히려 악화되었습니다. 이는 해당 손가락들의 압력 패턴이 이미지보다 센서 신호에 더 강하게 의존하거나, 데이터 내에서 손가락별 압력 분포가 불균형하여 이미지 특징을 결합하는 과정에서 일부 출력 차원이 덜 안정적으로 학습되었기 때문일 수 있습니다. 따라서 multi-output 모델에서는 전체 평균 성능뿐 아니라 손가락별 성능을 함께 확인하는 것이 중요합니다.

### 10.3 과일별 성능 비교

| 과일 | 샘플 수 | Tabular Frame MAE | Multimodal Frame MAE | 해석 |
|---|---:|---:|---:|---|
| 01.사과 | 81 | 0.3005 | 0.3052 | 소폭 악화 |
| 02.배(수확) | 25 | 0.2233 | 0.2154 | 개선 |
| 04.복숭아 | 29 | 0.3303 | 0.3049 | 개선 |
| 05.자두 | 27 | 0.3083 | 0.2817 | 개선 |

과일별 평균 MAE를 보면 메인 모델은 배(수확), 복숭아, 자두에서 베이스라인보다 낮은 오차를 보였습니다. 특히 복숭아와 자두에서는 이미지 정보가 과일의 외형, 크기, 위치, 접촉 장면과 관련된 추가 단서를 제공하여 압력 예측에 도움을 준 것으로 볼 수 있습니다.

반면 사과에서는 멀티모달 모델의 frame MAE가 0.3005에서 0.3052로 소폭 증가했습니다. 사과 샘플 수가 가장 많음에도 개선이 제한적이었다는 점은, 현재 이미지 입력이 사과 압력 예측에 충분한 추가 정보를 제공하지 못했거나 사과 데이터에서는 센서 기반 특징만으로도 대부분의 압력 변화가 설명되었을 가능성을 보여줍니다.

### 10.4 학습 과정 분석

Tabular Only MLP는 50 epoch까지 학습되었고 validation MSE 최저값은 50 epoch에서 0.1382였습니다. 즉 베이스라인은 학습이 비교적 안정적으로 진행되었으며, 지정한 epoch 내에서 validation loss가 계속 완만하게 개선되었습니다.

Late Fusion ResNet-18은 17 epoch에서 EarlyStopping이 발생했고, validation MSE 최저값은 9 epoch에서 0.1346이었습니다. 메인 모델은 베이스라인보다 더 낮은 최저 validation MSE를 달성했지만, 이후 validation loss가 다시 증가하여 조기 종료되었습니다. 이는 pretrained ResNet-18과 regression head를 함께 사용하는 모델이 더 높은 표현력을 갖지만, 데이터 수에 비해 모델 용량이 커 과적합이 더 빠르게 나타날 수 있음을 의미합니다.

따라서 차등 학습률 fine-tuning과 EarlyStopping은 본 실험에서 필수적인 안정화 전략으로 작용했습니다. Backbone에는 작은 학습률(`1e-5`)을 적용해 사전학습된 이미지 특징을 보존하고, regression head에는 상대적으로 큰 학습률(`1e-3`)을 적용해 새로운 5출력 회귀 문제에 빠르게 적응하도록 했습니다.

### 10.5 Failure Analysis

Late Fusion ResNet-18에서 5개 손가락 평균 MSE가 가장 큰 상위 5개 프레임은 다음과 같습니다.

| 순위 | 과일 | 이미지 파일 | Frame MSE | 실제 압력값 | 예측 압력값 | 주요 오차 |
|---:|---|---|---:|---|---|---|
| 1 | 04.복숭아 | `250814_PEAC_0209_004_R.jpeg` | 0.6317 | [1.7, 2.0, 2.2, 1.0, 0.4] | [1.778, 1.858, 0.652, 0.182, 0.138] | Finger 3, 4 과소예측 |
| 2 | 05.자두 | `250715_PLUM_0437_003_R.jpeg` | 0.4702 | [3.1, 1.9, 0.1, 0.2, 0.1] | [2.542, 2.452, 1.275, 0.704, 0.417] | Finger 3, 4, 5 과대예측 |
| 3 | 04.복숭아 | `250821_PEAC_0495_004_R.jpeg` | 0.4267 | [1.7, 1.7, 2.2, 0.6, 0.6] | [2.279, 2.239, 1.000, 0.547, 0.348] | Finger 3 과소예측 |
| 4 | 04.복숭아 | `250821_PEAC_0595_003_R.jpeg` | 0.3961 | [2.9, 1.8, 1.3, 0.5, 0.4] | [1.733, 1.784, 0.619, 0.159, 0.205] | Finger 1, 3 과소예측 |
| 5 | 04.복숭아 | `250821_PEAC_0649_004_R.jpeg` | 0.3913 | [1.3, 2.6, 2.1, 0.8, 0.4] | [2.040, 2.080, 1.078, 0.505, 0.316] | Finger 1 과대예측, Finger 3 과소예측 |

실패 케이스의 공통점은 Finger 3의 오차가 매우 크게 나타난다는 점입니다. 상위 5개 실패 프레임 중 대부분에서 Finger 3이 실제보다 크게 낮거나 높게 예측되었습니다. 이는 손가락별 압력 분포가 서로 다르며, 특히 Finger 3의 압력 패턴이 현재 모델 구조에서 안정적으로 학습되지 않았음을 시사합니다.

또한 실패 케이스가 복숭아와 자두에 집중되어 있습니다. 이 두 과일은 과일 표면, 크기, 수확 자세, 접촉 위치 변화가 상대적으로 다양할 수 있으며, 그 결과 같은 과일 종류 내에서도 손가락별 압력 패턴의 분산이 커졌을 가능성이 있습니다.

## 11. 결론

본 실험에서 Late Fusion ResNet-18 기반 멀티모달 모델은 정형 센서 데이터만 사용하는 Tabular Only MLP보다 전체 RMSE와 MAE를 각각 약 1.70%, 2.66% 개선했습니다. 이는 이미지 정보가 센서 데이터만으로는 설명하기 어려운 수확 장면의 시각적 맥락을 보완하여 5개 손가락 압력의 평균적인 회귀 오차를 줄였다는 점에서 의미가 있습니다.

그러나 Perfect Match Accuracy는 감소했으며, Finger 3과 Finger 4에서는 손가락별 오차가 악화되었습니다. 따라서 현재 메인 모델은 전체 평균 오차를 줄이는 데는 효과적이지만, 모든 손가락을 동시에 정밀하게 맞히는 수준에는 도달하지 못했습니다.

종합하면, 이미지 기반 late fusion은 multi-output 손가락 압력 예측에서 유효한 방향이지만, 손가락별 출력의 불균형과 특정 과일/특정 손가락에서의 실패 패턴을 해결하기 위한 추가 개선이 필요합니다.

## 12. 향후 개선 방향

- 손가락별 압력 분포를 분석하고 고압력/저빈도 구간에 weighted loss 적용
- `MSELoss` 대신 `HuberLoss` 또는 손가락별 가중 MSE 비교
- pressure bin 기반 weighted sampling으로 극단 압력 프레임 보강
- ResNet backbone 일부 layer freeze 후 점진적 unfreeze 실험
- 이미지 crop 또는 attention 기반 접촉 영역 강조
- 손가락별 metric을 기준으로 취약 손가락에 대한 ablation study 수행
