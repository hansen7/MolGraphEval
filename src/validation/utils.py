from typing import Dict, List, Optional, Union

import pandas as pd

from config import Config
from config.training_config import TrainingConfig
from config.validation_config import ValidationConfig
from datasets import MoleculeDataset
from splitters import random_scaffold_split, random_split, scaffold_split


def get_dataset_extraction(
    config: Union[TrainingConfig, ValidationConfig]
) -> MoleculeDataset:
    root: str = f"data/molecule_datasets/{config.dataset}/"
    return MoleculeDataset(root=root, dataset=config.dataset)


def get_dataset_split(
    config: Union[TrainingConfig, ValidationConfig],
    dataset: MoleculeDataset,
    smiles_list: Optional[List[str]],
) -> Dict[str, MoleculeDataset]:
    if config.split == "scaffold":
        train_, val_, test_, smile_ = scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            return_smiles=True,
        )
    elif config.split == "random":
        train_, val_, test_, smile_ = random_split(
            dataset,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=config.seed,
            smiles_list=smiles_list,
        )
    elif config.split == "random_scaffold":
        train_, val_, test_, smile_ = random_scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=config.seed,
            return_smiles=True,
        )
    else:
        raise ValueError("Invalid split option.")
    return {
        "train": train_,
        "val": val_,
        "test": test_,
        "smiles": smile_,
    }


def get_smiles_list(config: Config) -> List[str]:
    smiles_list = pd.read_csv(
        f"data/molecule_datasets/{config.dataset}/processed/smiles.csv",
        header=None,
    )
    return smiles_list[0].tolist()
