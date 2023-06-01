from typing import Union

from .training_config import TrainingConfig
from .validation_config import ValidationConfig


def aug_whitening(
    config: Union[TrainingConfig, ValidationConfig]
) -> Union[TrainingConfig, ValidationConfig]:
    """Remove the augmentations in dataloader."""

    """ AM PreTrainer """
    config.mask_rate = 0.0
    config.mask_edge = 0

    """ GraphCL/JOAO/JOAOv2 PreTrainer """
    config.aug_mode = "no_aug"
    config.aug_strength = 0.0
    config.aug_prob = 0.0

    # unresolved Methods:
    # GraphGPTGNN, Contextual

    # unchanged Methods:
    # Motif, IM
    return config
