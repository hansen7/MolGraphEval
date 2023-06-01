import torch

from config.training_config import TrainingConfig, parse_config
from datasets import MoleculeDataset
from init import (
    get_data_loader,
    get_dataset,
    get_device,
    get_model,
    get_optimizer,
    get_pretrainer,
    init,
)
from load_save import save_model
from logger import CombinedLogger


def pretrain_model(
    config: TrainingConfig,
    dataset: MoleculeDataset,
    model: torch.nn.Module,
    device: torch.device,
) -> None:

    """=== Generic Pre-Training Wrapper ==="""
    logger = CombinedLogger(config=config)
    optimizer = get_optimizer(config=config, model=model)
    train_data_loader = get_data_loader(config=config, dataset=dataset)
    pre_trainer = get_pretrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        device=device,
        logger=logger,
    )

    for epoch in range(config.epochs):
        # the epoch_0 is random initliased weights
        if epoch % config.epochs_save == 0 and config.save_model:
            save_model(config=config, model=pre_trainer.model, epoch=epoch)
        if config.pretrainer == "RGCL":
            pre_trainer.train_for_one_epoch(dataset)
        else:
            pre_trainer.train_for_one_epoch(train_data_loader)

    if config.save_model:
        save_model(config=config, model=pre_trainer.model, epoch=epoch)


def main() -> None:
    config = parse_config()
    init(config=config)
    device = get_device(config=config)
    dataset = get_dataset(config=config)
    model = get_model(config=config).to(device)
    pretrain_model(config=config, dataset=dataset, model=model, device=device)


if __name__ == "__main__":
    main()
