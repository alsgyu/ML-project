from __future__ import annotations

import random
import re
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """재현 가능한 실험을 위해 난수 시드를 고정한다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_finger_pressure(value: Any) -> list[float] | None:
    """'2.2, 1.8, 2.1, 0.8, 0.4' 형태의 문자열을 5개 float 리스트로 변환한다."""

    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(value))
    if len(numbers) != 5:
        return None
    return [float(number) for number in numbers]
