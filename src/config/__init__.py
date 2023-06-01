import dataclasses
from abc import ABC


def str2bool(v: str):
    return v.lower() == "true"


@dataclasses.dataclass
class Config(ABC):
    # about seed and basic info
    seed: int
    runseed: int
    device: int
    no_cuda: bool
    dataset: str
    # about model and pre-trainer
    model: str
    pretrainer: str
