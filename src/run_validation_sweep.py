import argparse
from datetime import datetime
from typing import Union

import yaml

import wandb

TIME_STR = "{:%Y_%m_%d_%H_%M_%S_%f}".format(datetime.now())
DATE_str = "{:%Y_%m_%d}".format(datetime.now())

PATH_TO_CONFIGS = "./config/sweeps/"


def parse_sweep_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val_task", type=str, default="prober", choices=["prober", "smoothing_metric"]
    )
    parser.add_argument(
        "--probe_task",
        type=str,
        default="node_degree",
        choices=[
            "node_degree",
            "node_centrality",
            "node_clustering",
            "graph_diameter",
            "link_prediction",
            "jaccard_coefficient",
            "katz_index",
            "graph_edit_distance",
            "cycle_basis",
        ],
    )
    parser.add_argument("--dataset", type=str, default="bace")
    parser.add_argument("--num_sweeps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def read_yaml(path: str) -> dict:
    with open(path, "r") as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded


def add_const_param(
    param_name: str, param_value: Union[str, int], config_dict: dict
) -> None:
    param_dict = {"distribution": "constant", "value": param_value}
    config_dict["parameters"][param_name] = param_dict


def add_name(name: str, config_dict: dict) -> None:
    config_dict["name"] = name


if __name__ == "__main__":
    args = parse_sweep_config()
    yaml_path = PATH_TO_CONFIGS + f"mlp.yaml"
    sweep_config = read_yaml(path=yaml_path)
    sweep_run_name = f"mlp-{args.seed}"
    add_name(sweep_run_name, sweep_config)
    add_const_param("val_task", args.val_task, sweep_config)
    add_const_param("dataset", args.dataset, sweep_config)
    add_const_param("device", args.device, sweep_config)
    add_const_param("seed", args.seed, sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="GraphPreTrainingBenchmarking")
    wandb.agent(sweep_id, count=args.num_sweeps)
