import pickle as pkl
from pathlib import Path
from typing import List

import torch
from torch_geometric.data.dataset import Dataset
from tqdm import tqdm

from config.training_config import TrainingConfig
from config.validation_config import ValidationConfig
from models.pre_trainer_model import PreTrainerModel

try:
    from torch_geometric.data.dataloader import DataLoader
except ModuleNotFoundError:
    from torch_geometric.loader.dataloader import DataLoader


def infer_and_save_embeddings(
    config: ValidationConfig,
    model: PreTrainerModel,
    device: torch.device,
    datasets: List[Dataset],
    loaders: List[DataLoader],
    smile_splits: list,
    save: bool = True,
) -> None:
    """Save graph and node representations for analysis.
    :param config: configurations
    :param model: pretrained or randomly-initialized model.
    :param device: device in use.
    :param datasets: train, valid and test data.
    :param loaders: train, valid and test data loaders.
    :param smile_splits: train, valid and test smiles.
    :param save: whether to save the pickle file.
    :return: None"""
    model.eval()
    for dataset, loader, smiles, split in zip(
        datasets, loaders, smile_splits, ["train", "valid", "test"]
    ):
        pbar = tqdm(total=len(loader))
        pbar.set_description(f"{split} embeddings extracted: ")
        graph_embeddings_list, node_embeddings_list = [], []
        for batch in loader:
            # if config.pretrainer == 'GraphCL' and isinstance(batch, list):
            #     batch = batch[0]  # remove the contrastive augmented data.
            batch = batch.to(device)
            with torch.no_grad():
                node_embeddings, graph_embeddings = model.get_embeddings(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
            unbatched_node_embedding = [[] for _ in range(batch.batch.max() + 1)]
            for embedding, graph_id in zip(node_embeddings, batch.batch):
                unbatched_node_embedding[graph_id].append(
                    embedding.detach().cpu().numpy()
                )
            graph_embeddings_list.append(graph_embeddings)
            node_embeddings_list.extend(unbatched_node_embedding)
            pbar.update(1)
        graph_embeddings_list = (
            torch.cat(graph_embeddings_list, dim=0).detach().cpu().numpy()
        )

        if save:
            save_embeddings(
                config=config,
                dataset=dataset,
                graph_embeddings=graph_embeddings_list,
                node_embeddings=node_embeddings_list,
                smiles=smiles,
                split=split,
            )

        pbar.close()
    # return dataset, graph_embeddings, node_embeddings, smiles


def save_embeddings(
    config: ValidationConfig,
    dataset: Dataset,
    graph_embeddings: List[torch.Tensor],
    node_embeddings: List[List[torch.Tensor]],
    smiles: List[str],
    split: str,
):
    Path(config.embedding_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{config.embedding_dir}{config.dataset}_{split}.pkl", "wb") as f:
        pkl.dump([graph_embeddings, node_embeddings, smiles], f)


def save_model(config: TrainingConfig, model: torch.nn.Module, epoch: int) -> None:
    saver_dict = {"model": model.state_dict()}
    cfg = pretrain_config(config)

    # TODO: Update this for G-Contextual,
    path_ = f"{config.output_model_dir}/{config.pretrainer}/{config.dataset}/{cfg}"
    Path(path_).mkdir(parents=True, exist_ok=True)
    torch.save(saver_dict, f"{path_}/epoch{epoch}_model_complete.pth")


def pretrain_config(config: TrainingConfig) -> None:
    cfg = ""
    if config.pretrainer == "AM":
        cfg = f"mask_rate-{config.mask_rate}"
    # TODO: run these experiments
    # elif config.pretrainer == "CP":
    # cfg = f"acsize-{config.csize}_atom_vocab_size-{config.atom_vocab_size}_contextpred_neg_samples-{config.contextpred_neg_samples}"
    elif config.pretrainer == "GraphCL":
        cfg = f"aug_mode-{config.aug_mode}_aug_strength-{config.aug_strength}_aug_prob-{config.aug_prob}"
    elif config.pretrainer in ["JOAO", "JOAOv2"]:
        cfg = f"gamma_joao-{config.gamma_joao}_gamma_joaov2-{config.gamma_joaov2}"
    # TODO: run these experiments
    elif config.pretrainer == "GraphMVP":
        cfg = f"alpha2-{config.GMVP_alpha2}_temper-{config.GMVP_T}"
    cfg = f"{cfg}_seed-{config.seed}_lr-{config.lr}"
    return cfg


def load_checkpoint(config: ValidationConfig, device: torch.device) -> dict:
    print(f"\nLoad checkpoint from path {config.input_model_file}...")
    return torch.load(config.input_model_file, map_location=device)
