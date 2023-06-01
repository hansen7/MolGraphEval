import argparse
import dataclasses

from config import Config, str2bool

# TODO (Jean): There are lots of redundancies between TrainingConfig and ValidationConfig
# I am sure there is a better way; maybe making ValidationConfig a subclass of TrainingConfig


@dataclasses.dataclass
class ValidationConfig(Config):
    # validation task
    val_task: str
    probe_task: str
    # logging
    log_filepath: str
    log_to_wandb: bool
    log_interval: int
    val_interval: int
    project_name: str
    run_name: str
    # about if we would print out eval metric for training data
    eval_train: bool
    input_data_dir: str
    # about loading and saving
    save_model: bool
    input_model_file: str
    output_model_dir: str
    embedding_dir: str
    verbose: bool
    # about dataset and dataloader
    batch_size: int
    num_workers: str
    # about molecule GNN
    gnn_type: str
    num_layer: int
    emb_dim: int
    dropout_ratio: float
    graph_pooling: str
    JK: str
    # gnn_lr_scale: float
    aggr: str

    # for ProberTaskMLP
    mlp_dim_hidden: int
    mlp_dim_out: int
    mlp_num_layers: int
    mlp_batch_norm: bool
    mlp_initializer: str
    mlp_dropout: float
    mlp_activation: str
    mlp_leaky_relu: float

    # for GraphCL
    aug_mode: str
    aug_strength: float
    aug_prob: float
    # for AttributeMask
    mask_rate: float
    mask_edge: int
    num_atom_type: int
    num_edge_type: int
    # for ContextPred
    csize: int
    atom_vocab_size: int
    contextpred_neg_samples: int
    # for JOAO
    gamma_joao: float
    gamma_joaov2: float

    # Validation metric arguments

    # ProberTask
    optimizer_name: str
    split: str
    batch_size: int
    epochs: int
    lr: float
    # lr_scale: float
    weight_decay: float
    criterion_type: float


def parse_config(parser: argparse.ArgumentParser = None) -> ValidationConfig:
    parser = argparse.ArgumentParser() if parser is None else parser
    # val and probe task
    parser.add_argument(
        "--val_task",
        type=str,
        default="prober",
        choices=["prober", "smoothing_metric", "finetune"],
    )
    parser.add_argument("--probe_task", type=str, default="downstream")

    # seed and basic info
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runseed", type=int, default=0)
    parser.add_argument("--no_cuda", type=str2bool, default=False)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--model", type=str, default="gnn", choices=["gnn", "schnet", "egnn"]
    )
    parser.add_argument(
        "--pretrainer",
        type=str,
        default="AM",
        choices=[
            "Motif",
            "Contextual",
            "GPT_GNN",
            "GraphCL",
            "JOAO",
            "JOAOv2",
            "AM",
            "IM",
            "GraphMVP",
            "CP",
            "EP",
            "GraphMAE",
            "RGCL",
            "FineTune",
        ],
    )

    # logging
    parser.add_argument("--log_filepath", type=str, default="./log/")
    parser.add_argument("--log_to_wandb", type=str2bool, default=True)
    parser.add_argument("--log_interval", default=10, type=int, help="Log steps")
    parser.add_argument(
        "--val_interval",
        default=1,
        type=int,
        help="Evaluate validation push_loss every n steps",
    )
    parser.add_argument(
        "--project_name", default="GraphEval", type=str, help="project name in wandb"
    )
    parser.add_argument("--run_name", type=str, help="run name in wandb")

    # about loading and saving
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--input_model_file", type=str, default="")
    parser.add_argument("--output_model_dir", type=str, default="")
    parser.add_argument("--embedding_dir", type=str, default="")
    # parser.add_argument("--embedding_dir", type=str,
    # default="./embedding_dir_x/Contextual/geom2d_nmol50000_nconf1_nupper1000/")
    parser.add_argument("--verbose", type=str2bool, default=False)

    # about dataset and dataloader
    parser.add_argument("--dataset", type=str, default="tox21")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)

    # ProberTask
    # about training strategies
    parser.add_argument("--optimizer_name", type=str, default="adam")
    parser.add_argument("--split", type=str, default="scaffold")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--criterion_type", type=str, default="mse")

    # about molecule GNN
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--graph_pooling", type=str, default="mean")
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        choices=["last", "sum", "max", "concat"],
        help="how the node features across layers are combined.",
    )
    # parser.add_argument("--gnn_lr_scale", type=float, default=1)
    parser.add_argument("--aggr", type=str, default="add")

    # for ProberTaskMLP
    parser.add_argument("--mlp_dim_hidden", type=int, default=600)
    parser.add_argument("--mlp_dim_out", type=int, default=1)
    parser.add_argument("--mlp_num_layers", type=int, default=2)
    parser.add_argument("--mlp_batch_norm", type=str2bool, default=False)
    parser.add_argument("--mlp_initializer", type=str, default="xavier")
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument(
        "--mlp_activation",
        type=str,
        default="relu",
        choices=["leaky_relu", "rrelu", "relu", "elu", "gelu", "prelu", "selu"],
    )
    parser.add_argument("--mlp_leaky_relu", type=float, default=0.5)

    # for GraphCL
    parser.add_argument("--aug_mode", type=str, default="sample")
    parser.add_argument("--aug_strength", type=float, default=0.2)
    parser.add_argument("--aug_prob", type=float, default=0.1)

    # for AttributeMask
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--mask_edge", type=int, default=0)
    parser.add_argument("--num_atom_type", type=int, default=119)
    parser.add_argument("--num_edge_type", type=int, default=5)

    # PreTrainer: G-Cont, will automatically adjust based on pre-training data
    parser.add_argument("--atom_vocab_size", type=int, default=1)

    # for ContextPred
    parser.add_argument("--csize", type=int, default=3)
    parser.add_argument("--contextpred_neg_samples", type=int, default=1)

    # for JOAO
    parser.add_argument("--gamma_joao", type=float, default=0.1)
    parser.add_argument("--gamma_joaov2", type=float, default=0.1)

    # about if we would print out eval metric for training data
    parser.add_argument("--eval_train", type=str2bool, default=True)
    parser.add_argument("--input_data_dir", type=str, default="")

    args = parser.parse_args()
    return ValidationConfig(**vars(args))
