import argparse
import dataclasses

from config import Config, str2bool


@dataclasses.dataclass
class TrainingConfig(Config):
    model: str
    pretrainer: str
    log_filepath: str
    log_to_wandb: bool
    log_interval: int
    project_name: str
    run_name: str
    val_interval: int
    eval_train: bool
    input_data_dir: str
    save_model: bool
    input_model_file: str
    output_model_dir: str
    verbose: bool
    num_workers: str
    optimizer_name: str
    split: str
    batch_size: int
    epochs: int
    epochs_save: int
    lr: float
    # lr_scale: float
    weight_decay: float
    gnn_type: str
    num_layer: int
    emb_dim: int
    dropout_ratio: float
    graph_pooling: str
    JK: str
    # gnn_lr_scale: float
    aggr: str

    # === GraphCL ===
    aug_mode: str
    aug_strength: float
    aug_prob: float
    # === AttrMask ===
    mask_rate: float
    mask_edge: int
    num_atom_type: int
    num_edge_type: int
    # === ContextPred ===
    csize: int
    contextpred_neg_samples: int
    atom_vocab_size: int
    # === JOAO ===
    gamma_joao: float
    gamma_joaov2: float
    # === GraphMVP ===
    GMVP_alpha1: float
    GMVP_alpha2: float
    GMVP_T: float
    GMVP_normalize: bool
    GMVP_CL_similarity_metric: str
    GMVP_CL_Neg_Samples: int
    GMVP_Masking_Ratio: float


def parse_config() -> TrainingConfig:
    parser = argparse.ArgumentParser()

    # Seed and Basic Info
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runseed", type=int, default=0)
    parser.add_argument("--no_cuda", type=str2bool, default=False, help="Disable CUDA")
    parser.add_argument(
        "--model", type=str, default="gnn", choices=["gnn", "schnet", "egnn"]
    )
    parser.add_argument(
        "--pretrainer",
        type=str,
        default="GraphCL",
        choices=[
            "Motif",
            "Contextual",
            "GPT_GNN",
            "GraphCL",
            "JOAO",
            "JOAOv2",
            "AM",
            "IM",
            "CP",
            "EP",
            "GraphMVP",
            "RGCL",
            "GraphMAE",
        ],
    )

    # Logging
    parser.add_argument("--log_filepath", type=str, default="./log/")
    parser.add_argument("--log_to_wandb", type=str2bool, default=True)
    parser.add_argument(
        "--log_interval", default=10, type=int, help="Log every n steps"
    )
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
    parser.add_argument("--verbose", type=str2bool, default=False)

    # about if we would print out eval metric for training data
    parser.add_argument("--eval_train", type=str2bool, default=True)
    parser.add_argument("--input_data_dir", type=str, default="")

    # Loading and saving model checkpoints
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--input_model_file", type=str, default="")
    parser.add_argument("--output_model_dir", type=str, default="./saved_models/")

    # about dataset and dataloader
    parser.add_argument("--dataset", type=str, default="bace")
    parser.add_argument("--num_workers", type=int, default=8)

    # Training strategies (shared by PreTraining and FineTuning)
    parser.add_argument("--optimizer_name", type=str, default="adam")
    parser.add_argument("--split", type=str, default="scaffold")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epochs_save", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0)

    # Molecule GNN
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

    # PreTrainer: GraphCL, JOAO, JOAOv2
    parser.add_argument("--aug_mode", type=str, default="sample")
    parser.add_argument("--aug_strength", type=float, default=0.2)
    parser.add_argument("--aug_prob", type=float, default=0.1)

    # PreTrainer: AttrMask
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--mask_edge", type=int, default=0)
    parser.add_argument("--num_atom_type", type=int, default=119)
    parser.add_argument("--num_edge_type", type=int, default=5)

    # PreTrainer: G-Cont, will automatically adjust based on pre-training data
    parser.add_argument("--atom_vocab_size", type=int, default=1)

    # PreTrainer: ContextPred
    parser.add_argument("--csize", type=int, default=3)
    parser.add_argument("--contextpred_neg_samples", type=int, default=1)

    # PreTrainer: JOAO and JOAOv2
    parser.add_argument("--gamma_joao", type=float, default=0.1)
    parser.add_argument("--gamma_joaov2", type=float, default=0.1)

    # PreTrainer: GraphMVP
    # Ref: https://github.com/chao1224/GraphMVP/blob/main/scripts_classification/submit_pre_training_GraphMVP_hybrid.sh#L14-L50
    parser.add_argument("--GMVP_alpha1", type=float, default=1)
    parser.add_argument("--GMVP_alpha2", type=float, default=1)  # 0.1, 1, 10
    parser.add_argument("--GMVP_T", type=float, default=0.1)  # 0.1, 0.2, 0.5, 1, 2
    parser.add_argument("--GMVP_normalize", type=bool, default=True)
    parser.add_argument(
        "--GMVP_CL_similarity_metric",
        type=str,
        default="EBM_dot_prod",
        choices=["InfoNCE_dot_prod", "EBM_dot_prod"],
    )
    parser.add_argument("--GMVP_CL_Neg_Samples", type=int, default=5)
    parser.add_argument("--GMVP_Masking_Ratio", type=float, default=0.0)

    args = parser.parse_args()
    return TrainingConfig(**vars(args))
