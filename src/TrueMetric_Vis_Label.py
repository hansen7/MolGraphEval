import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from config.validation_config import ValidationConfig
from init import get_prober_dataset, init


def tsne_from_embedding(dataset, probe_task, just_labels=False):

    config = ValidationConfig(
        seed=42,
        runseed=0,
        device=0,
        no_cuda=False,
        dataset=dataset,
        model="gnn",
        pretrainer="GraphCL",
        val_task="prober",
        probe_task=probe_task,
        log_filepath="./log/",
        log_to_wandb=False,
        log_interval=10,
        val_interval=1,
        eval_train=True,
        input_data_dir="",
        save_model=True,
        input_model_file="",
        output_model_dir="",
        embedding_dir="./embedding_dir_x/Contextual/geom2d_nmol50000_nconf1_nupper1000/",
        verbose=False,
        batch_size=256,
        num_workers=8,
        criterion_type="mse",
        gnn_type="gin",
        num_layer=5,
        emb_dim=300,
        dropout_ratio=0.5,
        graph_pooling="mean",
        JK="last",
        gnn_lr_scale=1,
        aggr="add",
        mlp_dim_hidden=600,
        mlp_num_layers=2,
        mlp_dim_out=1,
        mlp_batch_norm=False,
        mlp_initializer="xavier",
        mlp_dropout=0.0,
        mlp_activation="relu",
        mlp_leaky_relu=0.5,
        aug_mode="sample",
        aug_strength=0.2,
        aug_prob=0.1,
        mask_rate=0.15,
        mask_edge=0,
        num_atom_type=119,
        num_edge_type=5,
        csize=3,
        atom_vocab_size=508,
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

    train_dataset, val_dataset, test_dataset, criterion_type = get_prober_dataset(
        config=config
    )

    labels, reprs = [], []
    for item in train_dataset:
        labels.append(item["label"])
        reprs.append(item["representation"])

    for item in val_dataset:
        labels.append(item["label"])
        reprs.append(item["representation"])

    for item in test_dataset:
        labels.append(item["label"])
        reprs.append(item["representation"])

    if just_labels:
        return labels, reprs, None

    reprs = np.array(reprs)
    reprs_embedded = TSNE(
        n_components=2, learning_rate="auto", init="random"
    ).fit_transform(reprs)

    return labels, reprs, reprs_embedded


dataset_list = ["bbbp", "tox21", "toxcast", "sider", "clintox", "muv", "hiv", "bace"]

RDKIT_fragments = [
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
]


RDKIT_fragments_valid = [
    "fr_epoxide",
    "fr_lactam",
    "fr_morpholine",
    "fr_oxazole",
    "fr_tetrazole",
    "fr_N_O",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_morpholine",
    "fr_piperdine",
    "fr_thiazole",
    "fr_thiophene",
    "fr_urea",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_azo",
    "fr_benzene",
    "fr_imidazole",
    "fr_imide",
    "fr_piperzine",
    "fr_pyridine",
]


if __name__ == "__main__":

    for frag in RDKIT_fragments_valid:
        # probe_task = 'RDKiTFragment_%s' % frag
        # print(probe_task)
        # for idx, dataset in enumerate(dataset_list):
        #     labels, _, _ = tsne_from_embedding(
        #         dataset, probe_task, just_labels=True)
        #     print(dataset, np.unique(labels))
        # print("==================\n\n\n\n\n\n\n\n\n\n\n")

        probe_task = "RDKiTFragment_%s" % frag
        fig, axs = plt.subplots(2, 4, figsize=(15, 6), facecolor="w", edgecolor="k")
        axs = axs.ravel()
        all_labels = []

        for idx, dataset in enumerate(dataset_list):
            labels, _, _ = tsne_from_embedding(dataset, probe_task, just_labels=True)
            all_labels.append(labels)

            axs[idx].hist(labels, align="mid", bins=10, rwidth=0.4)
            axs[idx].set_title("%s" % dataset)
            axs[idx].set_yscale("log")
            plt.tight_layout()

        plt.savefig("../figures/label_%s.pdf" % frag, dpi=600)
        plt.show()
