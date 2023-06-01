import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from config.validation_config import ValidationConfig
from init import get_prober_dataset, init


def tsne_from_embedding(dataset, probe_task, embedding_dir, just_labels=False):
    config = ValidationConfig(
        seed=42,
        runseed=0,
        device=0,
        no_cuda=True,
        dataset=dataset,
        model="gnn",
        pretrainer="GraphCL",
        val_task="prober",
        probe_task=probe_task,
        log_filepath="./log/",
        log_to_wandb=True,
        log_interval=10,
        val_interval=1,
        eval_train=True,
        input_data_dir="",
        save_model=True,
        input_model_file="",
        output_model_dir="",
        embedding_dir=embedding_dir,
        verbose=False,
        batch_size=256,
        num_workers=8,
        gnn_type="gin",
        num_layer=5,
        emb_dim=300,
        dropout_ratio=0.5,
        graph_pooling="mean",
        JK="last",
        gnn_lr_scale=1,
        aug_mode="sample",
        aug_strength=0.2,
        aug_prob=0.1,
        mask_rate=0.15,
        mask_edge=0,
        num_atom_type=119,
        num_edge_type=5,
        csize=3,
        contextpred_neg_samples=1,
        gamma_joao=0.1,
        gamma_joaov2=0.1,
        optimizer_name="adam",
        split="scaffold",
        epochs=100,
        lr=0.001,
        lr_scale=1,
        weight_decay=0,
    )

    init(config=config)
    train_data, val_data, test_data, _ = get_prober_dataset(config=config)

    labels, reprs = [], []
    for item in train_data:
        labels.append(item["label"])
        reprs.append(item["representation"])

    for item in val_data:
        labels.append(item["label"])
        reprs.append(item["representation"])

    for item in test_data:
        labels.append(item["label"])
        reprs.append(item["representation"])

    reprs = np.array(reprs)

    if just_labels:
        return labels, reprs, None

    reprs_embedded = TSNE(
        n_components=2, learning_rate="auto", init="random"
    ).fit_transform(reprs)

    return labels, reprs, reprs_embedded


if __name__ == "__main__":

    dataset_list = [
        "bbbp",
        "tox21",
        "toxcast",
        "sider",
        "clintox",
        "muv",
        "hiv",
        "bace",
    ]
    probe_task = "node_centrality"
    PreTrainData = "geom2d_nmol50000_nconf1_nupper1000"
    figsize = (4.8 * 3, 5)

    for dataset in dataset_list[5:]:
        embedding_dir = "./embedding_dir_1019/random/"
        labels_random1, _, reprs_emb_random1 = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        embedding_dir = "./embedding_dir_1019/random_2/"
        labels_random2, _, reprs_emb_random2 = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        embedding_dir = "./embedding_dir_1019/random_3/"
        labels_random3, _, reprs_emb_random3 = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        scatter = ax1.scatter(
            reprs_emb_random1[:, 0], reprs_emb_random1[:, 1], c=labels_random1
        )
        legend1 = ax1.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Random Init 1)",
            ncol=4
        )

        scatter = ax2.scatter(
            reprs_emb_random2[:, 0], reprs_emb_random2[:, 1], c=labels_random2
        )
        legend2 = ax2.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Random Init 2)",
            ncol=4
        )

        scatter = ax3.scatter(
            reprs_emb_random2[:, 0], reprs_emb_random2[:, 1], c=labels_random3
        )
        legend3 = ax3.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Random Init 3)",
            ncol=4
        )
        plt.tight_layout()

        plt.draw()
        fig.savefig(
            "../figures/tsne_NodeCentrality_%s1.png" % dataset,
            bbox_extra_artists=(legend1, legend2, legend3),
            bbox_inches="tight",
            dpi=200,
        )

        embedding_dir = "./embedding_dir_1019/AM/%s/" % PreTrainData
        labels_am, _, reprs_embedded_am = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        embedding_dir = "./embedding_dir_1019/AM/geom2d_nmol100000_nconf1_nupper1000/"
        labels_am10k, _, reprs_embedded_am_10k = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        embedding_dir = "./embedding_dir_1019/AM/geom2d_nmol200000_nconf1_nupper1000/"
        labels_am20k, _, reprs_embedded_am_20k = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        scatter = ax1.scatter(
            reprs_embedded_am[:, 0], reprs_embedded_am[:, 1], c=labels_am
        )
        legend1 = ax1.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Pre-Trained AM50k)",
            ncol=4
        )

        scatter = ax2.scatter(
            reprs_embedded_am_10k[:, 0], reprs_embedded_am_10k[:, 1], c=labels_am10k
        )
        legend2 = ax2.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Pre-Trained AM100k)",
            ncol=4
        )

        scatter = ax3.scatter(
            reprs_embedded_am_20k[:, 0], reprs_embedded_am_20k[:, 1], c=labels_am20k
        )
        legend3 = ax3.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Pre-Trained AM200k)",
            ncol=4
        )
        plt.tight_layout()

        plt.draw()
        fig.savefig(
            "../figures/tsne_NodeCentrality_%s2.png" % dataset,
            bbox_extra_artists=(legend1, legend2, legend3),
            bbox_inches="tight",
            dpi=200,
        )

        embedding_dir = "./embedding_dir_1019/IM/%s/" % PreTrainData
        labels_im, _, reprs_embedded_im = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        embedding_dir = "./embedding_dir_1019/IM/geom2d_nmol100000_nconf1_nupper1000/"
        labels_im10k, _, reprs_embedded_im_10k = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        embedding_dir = "./embedding_dir_1019/IM/geom2d_nmol200000_nconf1_nupper1000/"
        labels_im20k, _, reprs_embedded_im_20k = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        scatter = ax1.scatter(
            reprs_embedded_im[:, 0], reprs_embedded_im[:, 1], c=labels_im
        )
        legend1 = ax1.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Pre-Trained IM50k)",
            ncol=4
        )

        scatter = ax2.scatter(
            reprs_embedded_im_10k[:, 0], reprs_embedded_im_10k[:, 1], c=labels_im10k
        )
        legend2 = ax2.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Pre-Trained IM100k)",
            ncol=4
        )

        scatter = ax3.scatter(
            reprs_embedded_im_20k[:, 0], reprs_embedded_im_20k[:, 1], c=labels_im20k
        )
        legend3 = ax3.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Pre-Trained IM200k)",
            ncol=4
        )
        plt.tight_layout()
        plt.draw()
        fig.savefig(
            "../figures/tsne_NodeCentrality_%s3.png" % dataset,
            bbox_extra_artists=(legend1, legend2, legend3),
            bbox_inches="tight",
            dpi=200,
        )

        embedding_dir = "./embedding_dir_1019/GraphCL/%s/" % PreTrainData
        labels_cl, _, reprs_embedded_cl = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        embedding_dir = "./embedding_dir_1019/JOAO/%s/" % PreTrainData
        labels_joao, _, reprs_embedded_joao = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        embedding_dir = "./embedding_dir_1019/JOAOv2/%s/" % PreTrainData
        labels_joaov2, _, reprs_embedded_joaov2 = tsne_from_embedding(
            dataset, probe_task, embedding_dir
        )

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        scatter = ax1.scatter(
            reprs_embedded_cl[:, 0], reprs_embedded_cl[:, 1], c=labels_cl
        )
        legend1 = ax1.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Pre-Trained GraphCL)",
            ncol=4
        )

        scatter = ax2.scatter(
            reprs_embedded_joao[:, 0], reprs_embedded_joao[:, 1], c=labels_joao
        )
        legend2 = ax2.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Pre-Trained JOAO)",
            ncol=4
        )

        scatter = ax3.scatter(
            reprs_embedded_joaov2[:, 0], reprs_embedded_joaov2[:, 1], c=labels_joaov2
        )
        legend3 = ax3.legend(
            *scatter.legend_elements(),
            loc=(0.0, 1.02),
            title="Node Centrality (Pre-Trained JOAOv2)",
            ncol=4
        )
        plt.tight_layout()
        plt.draw()
        fig.savefig(
            "../figures/tsne_NodeCentrality_%s4.png" % dataset,
            bbox_extra_artists=(legend1, legend2, legend3),
            bbox_inches="tight",
            dpi=200,
        )
