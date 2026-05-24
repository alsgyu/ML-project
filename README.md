# 농산물 수확과정 로봇의 손가락별 파지 압력 예측

AI Hub의 **단계적 사고 기반 농작업 로봇 데이터셋**을 활용해 과일 수확 상황에서 이미지 입력에 대해 로봇 손가락 5개의 압력을 예측합니다.

RGB 이미지와 로봇 센서 데이터를 함께 학습에 사용해 5개 손가락 압력을 동시에 추정합니다.

목표는 각 과일마다 수확시 손상을 최대한 줄이는 로봇 파지 제어를 위해, 수확 장면의 센서 정보에 시각 정보를 결합했을 때 압력 예측 성능이 개선되는지 확인하는 것입니다. 

</br>

```text
y = [finger_1, finger_2, finger_3, finger_4, finger_5]
```

## Dataset

| 항목 | 내용 |
|---|---|
| 출처 | [AI Hub: 단계적 사고 기반 농작업 로봇 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71887) |
| 분야 | 농작업 로봇, Physical AI |
| 데이터 유형 | RGB 이미지, Depth 이미지, CSV 센서 데이터, JSON 라벨 |
| 구축 / 갱신 | 2025 / 2026-05 |
| 전체 규모 | 원천 데이터 90,582건, 라벨링 데이터 30,194건 |
| 현재 프로젝트 사용 범위 | Validation 수확 데이터의 RGB 이미지와 CSV 센서 데이터 |

---

전체 원본 데이터는 적화, 적과, 가지치기, 수확 작업 데이터를 포함하고 있습니다. 대규모 데이터셋 중 로컬 컴퓨터 환경의 학습 및 데이터 관리 용이성을 고려하여, Validation 서브셋(3GB) 내의 수확 데이터만를 추출하여 실험을 수행하였습니다.


<br>

## Model & Usage

베이스 모델과 메인 모델의 핵심 차이는 RGB 이미지를 학습에 함께 사용했는지 여부입니다.

### 학습 데이터 비교

| 역할 | 모델 이름 | 구현 이름 | 학습에 사용한 데이터 | 데이터 설명 | 예측 대상 |
|---|---|---|---|---|---|
| 베이스 모델 | 정형 데이터 MLP | `TabularOnlyMLP` | 정형 특징 | 센서 PCA 15차원 + 작물 종류 one-hot 5차원 | 5개 손가락 압력 |
| 메인 모델 | 이미지 결합 MLP | `LateFusionResNetRegressor` | 정형 특징 + RGB 이미지 | 베이스 모델 정형 특징 + ResNet-18 이미지 특징 512차원을 추가 | 5개 손가락 압력 |

정형 특징은 두 모델이 동일하게 사용합니다. 메인 모델은 RGB 이미지를 ResNet-18로 인코딩한 뒤, 그 이미지 특징을 정형 특징과 결합해 MLP regression head에 넣습니다.

</br>

---

### 전처리와 학습 설정

| 단계 | 내용 |
|---|---|
| 데이터 매칭 | CSV와 RGB 이미지를 파일명 stem 기준으로 연결 |
| 타겟 생성 | `finger_pressure` 문자열을 5개 float 값으로 변환 |
| 필터링 | 평균 압력이 `0.5` 미만인 프레임 제거 (초기 데이터에는 수확 전 손을 뻗는 장면이 포함되어 있기 때문)|
| 정형 특징 | median imputation, standardization, PCA 15차원 |
| 범주 특징 | 작물 종류 one-hot 5차원 |
| 이미지 | `224 x 224` resize, tensor 변환, ImageNet normalization |

`SimpleImputer`, `StandardScaler`, `PCA`는 데이터 누수를 막기 위해 train split에만 fit하고 validation/test에는 transform만 적용합니다.

---

### 데이터 분할

| Split | 샘플 수 |
|---|---:|
| Train | 486 |
| Validation | 162 |
| Test | 162 |

가능한 경우 작물 종류를 기준으로 stratified split을 적용합니다.

| 파라미터 그룹 | Learning rate |
|---|---:|
| ResNet-18 backbone | `1e-5` |
| Regression head | `1e-3` |

<br>

---

## Results

| 모델 | 학습에 사용한 데이터 | RMSE | MAE | Perfect Match |
|---|---|---:|---:|---:|
| 베이스 모델 | 정형 특징 | 0.3771 | 0.2952 | **3.09%** |
| 메인 모델 | 정형 특징 + RGB 이미지 | **0.3706** | **0.2874** | 1.23% |

---

메인 모델은 베이스 모델 대비 전체 RMSE를 1.70%, MAE를 2.66% 개선하며 평균적인 오차를 낮췄습니다. 하지만 5개 손가락의 예측 오차가 모두 0.2 이내여야 하는 엄격한 지표인 Perfect Match Accuracy는 3.09%에서 1.23%로 성능이 감소했습니다.

이러한 성능 감소의 원인은 아래 **각 손가락별 변화 수치**에서 도출할 수 있습니다. 메인 모델은 외부에 노출되어 시각적 단서가 명확한 Finger 1, 2, 5에서만 오차가 크게 줄어들며 성능이 개선되었습니다. (특히 Finger 2의 MAE는 8.57% 개선)

반면 시각 정보의 의존도나 기여도가 상대적으로 낮은 Finger 3, 4에서는 오차가 도리어 증가하는 불균형한 최적화 양상이 나타났습니다.

또한, 소규모 학습 데이터(486개) 대비 주입된 이미지 특징 공간(512차원)이 너무 고차원이었기 때문에 모델이 학습 과정에서 시각적 노이즈에 민감하게 반응하여 특정 출력 차원에 가중치 편향이 발생했을 가능성도 큽니다.

5개 채널 중 하나의 출력만 임계치를 벗어나도 프레임 전체가 실패로 처리되는 Perfect Match 지표의 특성상, 이처럼 Finger 3, 4에서 발생한 오차 증가가 전체 Perfect Match Accuracy의 하락의 결정적인 요인으로 분석됩니다.


<br>

### + 각 손가락별 변화 수치

| 손가락 | 베이스 RMSE | 메인 RMSE | 변화 | 베이스 MAE | 메인 MAE | 변화 |
|---|---:|---:|---:|---:|---:|---:|
| Finger 1 | 0.4917 | 0.4749 | **3.41% 개선** | 0.3959 | 0.3822 | **3.44% 개선** |
| Finger 2 | 0.4468 | 0.4158 | **6.93% 개선** | 0.3669 | 0.3355 | **8.57% 개선** |
| Finger 3 | 0.3989 | 0.4159 | 4.27% 악화 | 0.3266 | 0.3292 | 0.81% 악화 |
| Finger 4 | 0.2540 | 0.2717 | 6.97% 악화 | 0.2088 | 0.2217 | 6.14% 악화 |
| Finger 5 | 0.2143 | 0.2040 | **4.78% 개선** | 0.1781 | 0.1684 | **5.45% 개선** |

---

<br>

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 multimodal_finger_pressure.py --epochs 50 --batch-size 16
```

GPU 환경에서는 다음처럼 batch size와 worker 수를 늘릴 수 있습니다.

```bash
python3 multimodal_finger_pressure.py --epochs 50 --batch-size 32 --num-workers 4
```

<br>

