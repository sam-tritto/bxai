from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional
import numpy as np


class FeatureStatus(Enum):
    CONFIRMED = "Confirmed"
    REJECTED = "Rejected"
    TENTATIVE = "Tentative"


@dataclass
class FeaturePosterior:
    feature_name: str
    status: FeatureStatus
    mean: float
    std: float
    hdi_lower: float
    hdi_upper: float
    raw_parameters: Dict[str, Any]
