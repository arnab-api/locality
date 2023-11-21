from dataclasses import dataclass
from typing import Any, Optional, Union

from dataclasses_json import DataClassJsonMixin

from locality.dataset import VariableBindingFactRecallDataset


@dataclass(frozen=True)
class PredictedToken(DataClassJsonMixin):
    """A predicted token and its probability."""

    token: str
    token_id: int
    prob: float

    def __str__(self) -> str:
        return f"{self.token} (p={self.prob:.3f})"


@dataclass(frozen=True)
class SampleResult(DataClassJsonMixin):
    query: str
    answer: str
    prediction: list[PredictedToken]


@dataclass(frozen=True)
class TrialResult(DataClassJsonMixin):
    few_shot_demonstration: str
    samples: list[SampleResult]
    recall: list[float]


@dataclass(frozen=True)
class ExperimentResults(DataClassJsonMixin):
    experiment_specific_args: dict[str, Any]
    trial_results: list[TrialResult]
