from typing import Dict, List, Optional, Tuple, Union
import wandb
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.backends import cudnn as cudnn
from torch_geometric.loader import DataLoader

from config import Config
from config.training_config import TrainingConfig
from config.validation_config import ValidationConfig
from dataloader import (
    DataLoaderAE,
    DataLoaderMasking,
    DataLoaderSubstructContext,
    DataLoaderMaskingPred,
)
from datasets import (
    MoleculeDataset,
    MoleculeDataset_Contextual,
    MoleculeDataset_GPTGNN,
    MoleculeDataset_GraphCL,
    MoleculeDataset_RGCL,
    MoleculeDataset_Motif,
    Molecule3DMaskingDataset,
    Molecule3DDataset,
)
from logger import CombinedLogger
from models import (
    ContextPredictionModel,
    AttributeMaskingModel,
    EdgePredictionModel,
    ContextualModel,
    GraphMVPModel,
    GraphMAEModel,
    GraphCLModel,
    InfoMaxModel,
    JOAOv2Model,
    GPTGNNModel,
    MotifModel,
    RGCLModel,
    GraphPred,
    GNN,
)
from models.building_blocks.auto_encoder import AutoEncoder, VariationalAutoEncoder
from models.building_blocks.schnet import SchNet
from models.building_blocks.mlp import MLP

from models.pre_trainer_model import PreTrainerModel
from pretrainers import (
    ContextPredictionPreTrainer,
    AttributeMaskingPreTrainer,
    EdgePredictionPreTrainer,
    ContextualPreTrainer,
    GraphMVPPreTrainer,
    GraphMAEPreTrainer,
    GraphCLPreTrainer,
    InfoMaxPreTrainer,
    JOAOv2PreTrainer,
    GPTGNNPreTrainer,
    MotifPreTrainer,
    JOAOPreTrainer,
    RGCLPreTrainer,
    PreTrainer,
    # GraphPred,
)
from splitters import random_scaffold_split, random_split, scaffold_split
from util import ExtractSubstructureContextPair, MaskAtom, NegativeEdge
from validation.metrics.smoothing_metric import MeanAverageDistance
from validation.dataset import ProberDataset
from validation.task import Task
from validation.task.graph_level import (
    AverageClusteringCoefficientDataset,
    AssortativityCoefficientDataset,
    NodeConnectivityDataset,
    GraphDiameterDataset,
    RDKiTFragmentDataset,
    CycleBasisDataset,
    DownstreamDataset,
)
from validation.task.node_level import (
    NodeCentralityDataset,
    NodeClusteringDataset,
    NodeDegreeDataset,
)

# from validation.task.graph_edit_distance import GraphEditDistanceDataset
from validation.task.pair_level import (
    JaccardCoefficientDataset,
    LinkPredictionDataset,
    KatzIndexDataset,
)
from validation.task.prober_task import ProberTask
from load_save import load_checkpoint


AVAILABLE_PRETRAINERS = {
    "AM": AttributeMaskingPreTrainer,
    "Contextual": ContextualPreTrainer,
    "CP": ContextPredictionPreTrainer,
    "GPT_GNN": GPTGNNPreTrainer,
    "GraphCL": GraphCLPreTrainer,
    "IM": InfoMaxPreTrainer,
    "JOAO": JOAOPreTrainer,
    "JOAOv2": JOAOv2PreTrainer,
    "Motif": MotifPreTrainer,
    "EP": EdgePredictionPreTrainer,
    "GraphMVP": GraphMVPPreTrainer,
    "RGCL": RGCLPreTrainer,
    "GraphMAE": GraphMAEPreTrainer,
}
from os.path import dirname, abspath, join

ROOT = dirname(dirname(abspath(__file__)))

# to solve dataloader crash of GPT_GNN Pre-Training
# Ref: https://github.com/facebookresearch/maskrcnn-benchmark/issues/103#issuecomment-785815218
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


def init(config: Config) -> None:
    if config.run_name is not None:
        run_name = config.run_name
    elif isinstance(config, ValidationConfig):
        run_name = f"{config.pretrainer}-{config.val_task}-{config.probe_task}-{config.dataset}-{str(config.seed)}"
    elif isinstance(config, TrainingConfig):
        run_name = f"{config.pretrainer}-{config.dataset}-{str(config.seed)}"
    else:
        raise ValueError
    wandb.init(project=config.project_name, name=run_name, reinit=True)
    wandb.config.update(config)
    # logger = create_logger(filepath=f"{config.log_filepath}{TIME_STR}.log")
    # logger.info(config)
    print(config)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True


def get_device(config: Config) -> Optional[torch.device]:
    if not config.no_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.device}")
        cudnn.deterministic = True
        cudnn.benchmark = True
        return device
    return torch.device("cpu")


def get_model(config: Union[TrainingConfig, ValidationConfig]) -> PreTrainerModel:
    # base_model = None
    if config.model == "gnn":  # used in embedding extraction
        model = GNN(config=config)
    else:
        raise NotImplementedError(f"Model {config.model} not implemented")

    if config.pretrainer in ["GraphCL", "JOAO"]:
        model = GraphCLModel(config=config, gnn=model)
    elif config.pretrainer == "GPT_GNN":
        model = GPTGNNModel(config=config, gnn=model)
    elif config.pretrainer == "JOAOv2":
        model = JOAOv2Model(config=config, gnn=model)
    elif config.pretrainer == "AM":
        model = AttributeMaskingModel(config=config, gnn=model)
    elif config.pretrainer == "Motif":
        model = MotifModel(config=config, gnn=model)
    elif config.pretrainer == "IM":
        model = InfoMaxModel(config=config, gnn=model)
    elif config.pretrainer == "Contextual":
        model = ContextualModel(config=config, gnn=model)
    elif config.pretrainer == "CP":
        model = ContextPredictionModel(config=config, gnn=model)
    elif config.pretrainer == "EP":
        model = EdgePredictionModel(config=config, gnn=model)
    elif config.pretrainer == "GraphMVP":
        ae2d3d = VariationalAutoEncoder(
            emb_dim=config.emb_dim, loss="l2", detach_target=True
        )
        ae3d2d = VariationalAutoEncoder(
            emb_dim=config.emb_dim, loss="l2", detach_target=True
        )
        gnn3d = SchNet(
            hidden_channels=config.emb_dim,
        )
        model = GraphMVPModel(
            config=config, gnn_2d=model, gnn_3d=gnn3d, ae2d3d=ae2d3d, ae3d2d=ae3d2d
        )
    elif config.pretrainer == "RGCL":
        model = RGCLModel(config=config, gnn=model)
    elif config.pretrainer == "GraphMAE":
        model = GraphMAEModel(config=config, gnn=model)
    if not model:
        raise NotImplementedError(f"Model {config.model} not implemented")
    return model


def get_smiles_list(config: Config) -> List[str]:
    smiles_list = pd.read_csv(
        f"data/molecule_datasets/{config.dataset}/processed/smiles.csv",
        header=None,
    )
    return smiles_list[0].tolist()


def get_dataset(config: Union[TrainingConfig, ValidationConfig]) -> MoleculeDataset:

    if "geom" in config.dataset:
        root = join(ROOT, f"data/GEOM/{config.dataset}/")
    else:
        root = join(ROOT, f"data/molecule_datasets/{config.dataset}/")

    if config.pretrainer == "Motif":
        dataset = MoleculeDataset_Motif(root=root, dataset=config.dataset)

    elif config.pretrainer == "Contextual":
        dataset = MoleculeDataset_Contextual(root=root, dataset=config.dataset)
        config.atom_vocab_size = len(dataset.atom_vocab)

    elif config.pretrainer == "AM":
        dataset = MoleculeDataset(
            root=root,
            dataset=config.dataset,
            transform=MaskAtom(
                num_atom_type=config.num_atom_type,
                num_edge_type=config.num_edge_type,
                mask_rate=config.mask_rate,
                mask_edge=config.mask_edge,
            ),
        )

    elif config.pretrainer == "EP":
        dataset = MoleculeDataset(
            root=root,
            dataset=config.dataset,
            transform=NegativeEdge(),
        )

    elif config.pretrainer == "IM":
        dataset = MoleculeDataset(root=root, dataset=config.dataset)

    elif config.pretrainer == "GPT_GNN":
        molecule_dataset = MoleculeDataset(root=root, dataset=config.dataset)
        dataset = MoleculeDataset_GPTGNN(molecule_dataset)

    elif config.pretrainer in ["GraphCL", "JOAO", "JOAOv2"]:
        dataset = MoleculeDataset_GraphCL(root=root, dataset=config.dataset)
        dataset.set_augMode(config.aug_mode)
        dataset.set_augStrength(config.aug_strength)
        dataset.set_augProb(np.ones(25) / 25)

    elif config.pretrainer == "CP":
        l1 = config.num_layer - 1
        l2 = l1 + config.csize
        dataset = MoleculeDataset(
            root=root,
            dataset=config.dataset,
            transform=ExtractSubstructureContextPair(
                k=config.num_layer,
                l1=l1,
                l2=l2,
            ),
        )

    elif config.pretrainer == "GraphMVP":
        dataset = Molecule3DDataset(
            root=root,
            dataset=config.dataset,
        )
        dataset = Molecule3DMaskingDataset(
            root=root, dataset=config.dataset, mask_ratio=config.GMVP_Masking_Ratio
        )

    elif config.pretrainer == "RGCL":
        dataset = MoleculeDataset_RGCL(root=root, dataset=config.dataset)

    elif config.pretrainer == "GraphMAE":
        dataset = MoleculeDataset(root=root, dataset=config.dataset)
    else:
        raise NotImplementedError(f"PreTrainer {config.pretrainer} not implemented")

    return dataset


def get_dataset_extraction(
    config: Union[TrainingConfig, ValidationConfig]
) -> MoleculeDataset:

    root: str = f"./data/molecule_datasets/{config.dataset}/"
    return MoleculeDataset(root=root, dataset=config.dataset)


def get_dataset_num(config: Union[TrainingConfig, ValidationConfig]) -> int:
    if config.dataset == "tox21":
        num_tasks = 12
    elif config.dataset == "hiv":
        num_tasks = 1
    elif config.dataset == "pcba":
        num_tasks = 128
    elif config.dataset == "muv":
        num_tasks = 17
    elif config.dataset == "bace":
        num_tasks = 1
    elif config.dataset == "bbbp":
        num_tasks = 1
    elif config.dataset == "toxcast":
        num_tasks = 617
    elif config.dataset == "sider":
        num_tasks = 27
    elif config.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")
    return num_tasks


def get_data_loader(
    config: Union[TrainingConfig, ValidationConfig],
    dataset: Union[MoleculeDataset, DataLoaderMasking],
    shuffle: bool = True,
) -> DataLoader:
    if config.pretrainer == "AM":
        return DataLoaderMasking(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
        )

    if config.pretrainer == "CP":
        return DataLoaderSubstructContext(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
        )

    if config.pretrainer == "EP":
        return DataLoaderAE(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
        )

    if config.pretrainer == "GraphMAE":
        return DataLoaderMaskingPred(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
        )

    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle,
    )


def get_data_loader_val(
    config: Union[TrainingConfig, ValidationConfig],
    dataset: MoleculeDataset,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle,
    )


def get_pretrainer(
    config: TrainingConfig,
    model: nn.Module,
    optimizer: Union[List, torch.optim.Optimizer],
    device: torch.device,
    logger: CombinedLogger,
) -> PreTrainer:
    if config.pretrainer not in AVAILABLE_PRETRAINERS:
        raise NotImplementedError(f"Pretrainer {config.pretrainer} not implemented")
    return AVAILABLE_PRETRAINERS[config.pretrainer](
        config=config,
        model=model,
        optimizer=optimizer,
        device=device,
        logger=logger,
    )


def get_optimizer(
    config: Union[TrainingConfig, ValidationConfig], model: nn.Module
) -> Union[List, torch.optim.Optimizer]:
    # default setting is adam, lr=0.001, weight_decay=0
    if config.optimizer_name == "adam" and config.pretrainer != "GraphMAE":
        return torch.optim.Adam(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_name == "adam" and config.pretrainer == "GraphMAE":
        if isinstance(config, TrainingConfig):
            return [
                torch.optim.Adam(
                    params=model.gnn.parameters(),
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                ),
                torch.optim.Adam(
                    model.atom_pred_decoder.parameters(),
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                ),
            ]
        else:
            return torch.optim.Adam(
                params=model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
    elif config.optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_name == "adagrad":
        return torch.optim.Adagrad(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_name == "sgd":
        return torch.optim.SGD(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer_name} unimplemented")


def get_prober_dataset(
    config: ValidationConfig,
) -> Tuple[ProberDataset, ProberDataset, ProberDataset, str]:

    criterion_type = "mse"  # default type

    # === Node Metric ===
    if config.probe_task == "node_degree":
        data_cls = NodeDegreeDataset
        # criterion_type = "ce"  # num_class: 11
    elif config.probe_task == "node_centrality":
        data_cls = NodeCentralityDataset
    elif config.probe_task == "node_clustering":
        data_cls = NodeClusteringDataset

    # === Pair Metric ===
    elif config.probe_task == "link_prediction":
        data_cls = LinkPredictionDataset
        # criterion_type = "bce"
    elif config.probe_task == "jaccard_coefficient":
        data_cls = JaccardCoefficientDataset
    elif config.probe_task == "katz_index":
        data_cls = KatzIndexDataset

    # === Graph Metric ===
    elif config.probe_task == "graph_diameter":
        data_cls = GraphDiameterDataset
    # elif config.probe_task == "graph_edit_distance":
    #     data_cls = GraphEditDistanceDataset
    #     criterion_type = 'mse'
    elif config.probe_task == "node_connectivity":
        data_cls = NodeConnectivityDataset
    elif config.probe_task == "cycle_basis":
        data_cls = CycleBasisDataset
        # criterion_type = "ce"
    elif config.probe_task == "assortativity_coefficient":
        data_cls = AssortativityCoefficientDataset
    elif config.probe_task == "average_clustering_coefficient":
        data_cls = AverageClusteringCoefficientDataset

    # === Substructure Metric ===
    elif "RDKiTFragment_" in config.probe_task:
        data_cls = RDKiTFragmentDataset

    # === Downstream Tasks ===
    elif config.probe_task == "downstream":
        data_cls = DownstreamDataset
        criterion_type = "bce"

    else:
        raise NotImplementedError

    train_dataset = data_cls(
        f"{config.embedding_dir}/{config.dataset}_train.pkl",
        config,
        "train",
        des=config.probe_task.replace("RDKiTFragment_", ""),
    ).create_datasets()
    val_dataset = data_cls(
        f"{config.embedding_dir}/{config.dataset}_valid.pkl",
        config,
        "val",
        des=config.probe_task.replace("RDKiTFragment_", ""),
    ).create_datasets()
    test_dataset = data_cls(
        f"{config.embedding_dir}/{config.dataset}_test.pkl",
        config,
        "test",
        des=config.probe_task.replace("RDKiTFragment_", ""),
    ).create_datasets()

    return train_dataset, val_dataset, test_dataset, criterion_type


def get_finetune_dataset(
    config: ValidationConfig,
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset, str]:

    criterion_type = "mse"  # default type

    # # === Node Metric ===
    # if config.probe_task == "node_degree":
    #     data_cls = NodeDegreeDataset
    #     # criterion_type = "ce"  # num_class: 11
    # elif config.probe_task == "node_centrality":
    #     data_cls = NodeCentralityDataset
    # elif config.probe_task == "node_clustering":
    #     data_cls = NodeClusteringDataset

    # # === Pair Metric ===
    # elif config.probe_task == "link_prediction":
    #     data_cls = LinkPredictionDataset
    #     criterion_type = "bce"
    # elif config.probe_task == "jaccard_coefficient":
    #     data_cls = JaccardCoefficientDataset
    # elif config.probe_task == "katz_index":
    #     data_cls = KatzIndexDataset

    # # === Graph Metric ===
    # elif config.probe_task == "graph_diameter":
    #     data_cls = GraphDiameterDataset
    # # elif config.probe_task == "graph_edit_distance":
    # #     data_cls = GraphEditDistanceDataset
    # #     criterion_type = 'mse'
    # elif config.probe_task == "node_connectivity":
    #     data_cls = NodeConnectivityDataset
    # elif config.probe_task == "cycle_basis":
    #     data_cls = CycleBasisDataset
    #     # criterion_type = "ce"
    # elif config.probe_task == "assortativity_coefficient":
    #     data_cls = AssortativityCoefficientDataset
    # elif config.probe_task == "average_clustering_coefficient":
    #     data_cls = AverageClusteringCoefficientDataset

    # # === Substructure Metric ===
    # elif "RDKiTFragment_" in config.probe_task:
    #     data_cls = RDKiTFragmentDataset

    # === Downstream Tasks ===
    if config.probe_task == "downstream":
        # data_cls = MoleculeDataset
        criterion_type = "bce"
        dataset = get_dataset_extraction(config=config)
        smiles_list = get_smiles_list(config=config)
        dataset_splits = get_dataset_split(
            config=config, dataset=dataset, smiles_list=smiles_list
        )
        # datasets_list = [dataset for dataset in dataset_splits.values()]
        return (
            dataset_splits["train"],
            dataset_splits["val"],
            dataset_splits["test"],
            criterion_type,
        )
    else:
        raise NotImplementedError

    # MoleculeDataset(root=root, dataset=config.dataset)
    # train_dataset = data_cls(
    #     f"{config.embedding_dir}{config.dataset}_train.pkl",
    #     config,
    #     "train",
    #     des=config.probe_task.replace("RDKiTFragment_", ""),
    # ).create_datasets()
    # val_dataset = data_cls(
    #     f"{config.embedding_dir}{config.dataset}_valid.pkl",
    #     config,
    #     "val",
    #     des=config.probe_task.replace("RDKiTFragment_", ""),
    # ).create_datasets()
    # test_dataset = data_cls(
    #     f"{config.embedding_dir}{config.dataset}_test.pkl",
    #     config,
    #     "test",
    #     des=config.probe_task.replace("RDKiTFragment_", ""),
    # ).create_datasets()

    # return train_dataset, val_dataset, test_dataset, criterion_type


def calculate_smoothing_metric(config):
    train_path = f"{config.embedding_dir}{config.dataset}_train.pkl"
    valid_path = f"{config.embedding_dir}{config.dataset}_valid.pkl"
    test_path = f"{config.embedding_dir}{config.dataset}_test.pkl"
    for idx, path in enumerate([train_path, valid_path, test_path]):
        metric = MeanAverageDistance(path, config, idx)
        metric.calculate_metric()


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


def get_probing_model(config: ValidationConfig, input_dim: int) -> nn.Module:
    assert config.val_task == "prober", "val_task should be finetune"
    mlp_dim_out = config.mlp_dim_out
    if config.probe_task == "downstream":
        mlp_dim_out = get_dataset_num(config)
    return nn.Linear(input_dim, mlp_dim_out)
    # return MLP(
    #         dim_input=input_dim,
    #         dim_hidden=config.mlp_dim_hidden,
    #         dim_output=mlp_dim_out,
    #         num_layers=config.mlp_num_layers,
    #         batch_norm=config.mlp_batch_norm,
    #         initializer=config.mlp_initializer,
    #         dropout=config.mlp_dropout,
    #         activation=config.mlp_activation,
    #         leaky_relu=config.mlp_leaky_relu,
    #         is_output_activation=False,
    #     )


def get_finetune_model(config: ValidationConfig, gnn: GNN) -> nn.Module:
    assert config.val_task == "finetune", "val_task should be finetune"
    num_tasks = config.mlp_dim_out
    if config.probe_task == "downstream":
        num_tasks = get_dataset_num(config)
    return GraphPred(config, gnn, num_tasks).to()


def get_task(config: ValidationConfig) -> Task:
    # logger = CombinedLogger(config=config)
    if config.val_task == "prober":
        device = get_device(config=config)
        train_dataset, val_dataset, test_dataset, criterion_type = get_prober_dataset(
            config=config
        )
        model = get_probing_model(config=config, input_dim=train_dataset.input_dim).to(
            device
        )
        optimizer = get_optimizer(config=config, model=model)
        return ProberTask(
            config=config,
            model=model,
            device=device,
            optimizer=optimizer,
            # logger=logger,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            criterion_type=criterion_type,
        )

    elif config.val_task == "finetune":
        device = get_device(config=config)

        train_dataset, val_dataset, test_dataset, criterion_type = get_finetune_dataset(
            config=config
        )
        pre_trained_model = get_model(config=config)
        checkpoint = load_checkpoint(config=config, device=device)

        model = get_finetune_model(config=config, gnn=pre_trained_model).to(device)
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer = get_optimizer(config=config, model=model)
        return ProberTask(
            config=config,
            model=model,
            device=device,
            # logger=logger,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            criterion_type=criterion_type,
        )

    elif config.val_task == "smoothing_metric":
        calculate_smoothing_metric(config)
    else:
        raise NotImplementedError("Unknown task {}".format(config.val_task))
