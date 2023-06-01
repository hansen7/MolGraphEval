from config.validation_config import parse_config
from init import (
    get_data_loader_val,
    get_dataset_extraction,
    get_dataset_split,
    get_device,
    get_model,
    get_smiles_list,
    init,
)
from load_save import infer_and_save_embeddings, load_checkpoint

# from validation.task import Task


def main() -> None:
    """=== Load the PreTrained Weights ==="""
    config = parse_config()
    init(config=config)
    device = get_device(config=config)

    """ === Set the Datasets and Loader === """
    dataset = get_dataset_extraction(config=config)
    smiles_list = get_smiles_list(config=config)
    dataset_splits = get_dataset_split(
        config=config, dataset=dataset, smiles_list=smiles_list
    )
    datasets_list = [dataset for dataset in dataset_splits.values()]
    # a list of MoleculeDataset class: ['train', 'val', 'test', 'smiles']
    loaders = [
        get_data_loader_val(config=config, dataset=dataset, shuffle=False)
        for dataset in datasets_list
    ]
    # a tuple of list of smiles: ('train', 'val', 'test')
    smile_splits = dataset_splits["smiles"]

    pre_trained_model = get_model(config=config).to(device)
    checkpoint = load_checkpoint(config=config, device=device)
    pre_trained_model.load_state_dict(checkpoint["model"])

    """ === Save Node & Graph Embeddings === """
    infer_and_save_embeddings(
        config=config,
        model=pre_trained_model,
        device=device,
        datasets=datasets_list,
        loaders=loaders,
        smile_splits=smile_splits,
    )


if __name__ == "__main__":
    main()
