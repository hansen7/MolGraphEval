# ref: github.com/snap-stanford/pretrain-gnns/blob/master/chems/loader.py
import json
import os
import pickle
import random
import sys
from itertools import chain, repeat
from os.path import join, abspath, dirname
import shutil
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

try:
    from .utils import (
        create_standardized_mol_id,
        get_largest_mol,
        mol_to_graph_data_obj_simple,
        mol_to_graph_data_obj_simple_3D,
        split_rdkit_mol_obj,
    )
except ImportError:
    from utils import (
        create_standardized_mol_id,
        get_largest_mol,
        mol_to_graph_data_obj_simple,
        mol_to_graph_data_obj_simple_3D,
        split_rdkit_mol_obj,
    )

# from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
# Ref: https://sourceforge.net/p/rdkit/mailman/message/30036309/
# lg = RDLogger.logger()
# lg.setLevel(RDLogger.ERROR)

BASE_ROOT = dirname(dirname(dirname(abspath(__file__))))
DOWNSTREAM_ROOT = join(BASE_ROOT, "data", "molecule_datasets")
GEOM_ROOT = join(BASE_ROOT, "data", "GEOM")
REMOVE_CACHE = True  # PyG 2.x -> PyG 1.x

sys.path.append(join(BASE_ROOT, "src"))
from splitters import scaffold_split


class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        empty=False,
    ):

        self.root = root
        self.empty = empty
        self.dataset = dataset
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        print()
        print("Dataset: ", self.dataset)
        super(MoleculeDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        if not self.empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            print(self.data)

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = (
            os.listdir(self.raw_dir) if os.path.exists(self.raw_dir) else list()
        )
        return file_name_list

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        return

    def process(self):
        def shared_extractor(smiles, rdkit_mol_objs, labels):
            """extract PyG Data objects and labels
            :param smiles: list or pd.Series
            :param rdkit_mol_objs: list
            :param labels: np.ndarray
            :return: two lists"""

            data_list = []
            if type(smiles) == pd.core.series.Series:
                smiles = smiles.to_list()
            print(
                "# SMILES: %5d  Labels: " % len(smiles),
                labels.shape,
                "  # Unique values: ",
                np.unique(labels).__len__(),
            )
            for i in tqdm(range(len(smiles))):
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i].reshape(1, -1))
                data_list.append(data)

            # if type(smiles) is not list:
            #     return data_list, smiles.to_list()
            return data_list, smiles

        if self.dataset == "zinc_standard_agent":
            data_list = []
            data_smiles_list = []
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path, sep=",", compression="gzip", dtype="str")
            zinc_id_list = list(input_df["zinc_id"])
            smiles_list = list(input_df["smiles"])

            for i in tqdm(range(len(smiles_list))):
                s = smiles_list[i]
                # each example contains a single species
                try:
                    rdkit_mol = AllChem.MolFromSmiles(s)
                    if rdkit_mol is not None:  # ignore invalid mol objects
                        # # convert aromatic bonds to double bonds
                        # Chem.SanitizeMol(rdkit_mol,
                        # sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        data = mol_to_graph_data_obj_simple(rdkit_mol)
                        # manually add mol id
                        id = int(zinc_id_list[i].split("ZINC")[1].lstrip("0"))
                        data.id = torch.tensor([id])
                        # id here is zinc id value,
                        # stripped of leading zeros
                        data_list.append(data)
                        data_smiles_list.append(smiles_list[i])
                except:
                    continue

        elif self.dataset == "chembl_filtered":
            data_list = []
            data_smiles_list = []
            downstream_dir = [
                "data/bace",
                "data/bbbp",
                "data/clintox",
                "data/esol",
                "data/freesolv",
                "data/hiv",
                # "data/lipophilicity",
                "data/muv",
                # 'dataset/pcba/processed/smiles.csv',
                "data/sider",
                "data/tox21",
                "data/toxcast",
            ]
            downstream_inchi_set = set()
            for path in downstream_dir:
                print(path)
                dataset_name = path.split("/")[1]
                downstream_dataset = MoleculeDataset(path, dataset=dataset_name)
                downstream_smiles = pd.read_csv(
                    join(path, "processed", "smiles.csv"), header=None
                )[0].tolist()

                assert len(downstream_dataset) == len(downstream_smiles)
                _, _, _, (train_smiles, valid_smiles, test_smiles) = scaffold_split(
                    downstream_dataset,
                    downstream_smiles,
                    task_idx=None,
                    null_value=0,
                    frac_train=0.8,
                    frac_valid=0.1,
                    frac_test=0.1,
                    return_smiles=True,
                )

                # remove both test and validation molecules
                remove_smiles = test_smiles + valid_smiles

                downstream_inchis = []
                for smiles in remove_smiles:
                    species_list = smiles.split(".")
                    for s in species_list:
                        # record inchi for all species, not just the largest
                        # (by default in create_standardized_mol_id if
                        # input has multiple species)
                        inchi = create_standardized_mol_id(s)
                        downstream_inchis.append(inchi)
                downstream_inchi_set.update(downstream_inchis)

            (
                smiles_list,
                rdkit_mol_objs,
                folds,
                labels,
            ) = _load_chembl_with_labels_dataset(join(self.root, "raw"))

            print("processing")
            for i in range(len(rdkit_mol_objs)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol is not None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    # sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        inchi = create_standardized_mol_id(smiles_list[i])
                        if inchi is not None and inchi not in downstream_inchi_set:
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id, which is index of
                            # the mol in the dataset
                            data.id = torch.tensor([i])
                            data.y = torch.tensor(labels[i, :])
                            # fold information
                            if i in folds[0]:
                                data.fold = torch.tensor([0])
                            elif i in folds[1]:
                                data.fold = torch.tensor([1])
                            else:
                                data.fold = torch.tensor([2])
                            data_list.append(data)
                            data_smiles_list.append(smiles_list[i])

        elif self.dataset == "tox21":
            smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "hiv":
            smiles_list, rdkit_mol_objs, labels = _load_hiv_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "bace":
            smiles_series, rdkit_mol_objs, _, labels = _load_bace_dataset(
                self.raw_paths[0]
            )
            smiles_list = smiles_series.to_list()
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "bbbp":
            smiles_list, rdkit_mol_objs, labels = _load_bbbp_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "clintox":
            smiles_list, rdkit_mol_objs, labels = _load_clintox_dataset(
                self.raw_paths[0]
            )
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "esol":
            smiles_list, rdkit_mol_objs, labels = _load_esol_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "freesolv":
            smiles_list, rdkit_mol_objs, labels = _load_freesolv_dataset(
                self.raw_paths[0]
            )
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "lipophilicity":
            smiles_list, rdkit_mol_objs, labels = _load_lipophilicity_dataset(
                self.raw_paths[0]
            )
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "muv":
            smiles_list, rdkit_mol_objs, labels = _load_muv_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "sider":
            smiles_list, rdkit_mol_objs, labels = _load_sider_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "toxcast":
            smiles_list, rdkit_mol_objs, labels = _load_toxcast_dataset(
                self.raw_paths[0]
            )
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels
            )

        elif self.dataset == "ptc_mr":
            input_path = self.raw_paths[0]
            data_list, data_smiles_list = [], []
            input_df = pd.read_csv(
                input_path, sep=",", header=None, names=["id", "label", "smiles"]
            )
            smiles_list = input_df["smiles"]
            labels = input_df["label"].values
            for i in range(len(smiles_list)):
                s = smiles_list[i]
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol is not None:
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    data.id = torch.tensor([i])
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == "mutag":
            data_list, data_smiles_list = [], []
            smiles_path = join(self.root, "raw", "mutag_188_data.can")
            labels_path = join(self.root, "raw", "mutag_188_target.txt")
            smiles_list = pd.read_csv(smiles_path, sep=" ", header=None)[0]
            labels = pd.read_csv(labels_path, header=None)[0].values
            for i in range(len(smiles_list)):
                s = smiles_list[i]
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol is not None:
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    data.id = torch.tensor([i])
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif "geom2d" in self.dataset:
            # todo: now we only work with 2D Graph SSL methods
            # in format of "geom2d_nmol%d_nconf%d_nupper%d"
            if not os.path.exists(self.processed_paths[0]):
                root_2d = "/".join(self.processed_paths[0].split("/")[:-2])
                root_3d = root_2d.replace("geom2d", "geom3d")
                cfg = self.processed_paths[0].split("/")[-3].split("_")
                n_mol, n_conf = (
                    int(cfg[1][4:]),
                    int(cfg[2][5:]),
                )
                GEOMDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf)
                GEOMDataset(
                    root=root_2d,
                    n_mol=n_mol,
                    n_conf=n_conf,
                    smiles_copy_from_3D_file="%s/processed/smiles.csv" % root_3d,
                )
            self.empty = False
            return

        else:
            raise NotImplementedError("Dataset {} not included.".format(self.dataset))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, "smiles.csv")
        if not os.path.exists(saver_path):
            print("saving to {}".format(saver_path))
            data_smiles_series.to_csv(saver_path, index=False, header=False)

        if not os.path.exists(self.processed_paths[0]):
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

        self.empty = False
        return


def _load_tox21_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]
    # convert 0 to -1, nan to 0
    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values)


def _load_hiv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df["HIV_active"]
    labels = labels.replace(0, -1)
    # convert 0 to -1, there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values.reshape(-1, 1))


def _load_bace_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["mol"], input_df["Class"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    # convert 0 to -1, assuming there are no nans
    labels = labels.replace(0, -1)
    folds = input_df["Model"]
    folds = folds.replace("Train", 0)  # 0 -> train
    folds = folds.replace("Valid", 1)  # 1 -> valid
    folds = folds.replace("Test", 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)

    return (
        smiles_list,
        rdkit_mol_objs_list,
        folds.values,
        labels.values.reshape(-1, 1),
    )


def _load_bbbp_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df["p_np"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [
        m if m is not None else None for m in rdkit_mol_objs_list
    ]
    preprocessed_smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None
        for m in preprocessed_rdkit_mol_objs_list
    ]

    # convert 0 to -1, there are no nans
    labels = labels.replace(0, -1)

    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)

    # drop NoneType, have this issue when re-generate with PyG 2.x
    non_loc = [i for (i, v) in enumerate(rdkit_mol_objs_list) if v is not None]
    sel_non = lambda a_list: [a_list[index] for index in non_loc]
    return (
        sel_non(preprocessed_smiles_list),
        sel_non(preprocessed_rdkit_mol_objs_list),
        labels.values[non_loc].reshape(-1, 1),
    )


def _load_clintox_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [
        m if m is not None else None for m in rdkit_mol_objs_list
    ]
    preprocessed_smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None
        for m in preprocessed_rdkit_mol_objs_list
    ]
    tasks = ["FDA_APPROVED", "CT_TOX"]
    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)

    non_loc = [i for (i, v) in enumerate(rdkit_mol_objs_list) if v is not None]
    sel_non = lambda a_list: [a_list[index] for index in non_loc]
    return (
        sel_non(preprocessed_smiles_list),
        sel_non(preprocessed_rdkit_mol_objs_list),
        labels.values[non_loc],
    )


def _load_esol_dataset(input_path):
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df["measured log solubility in mols per litre"]
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values.reshape(-1, 1))


def _load_freesolv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df["expt"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values.reshape(-1, 1))


def _load_lipophilicity_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df["exp"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values.reshape(-1, 1))


def _load_muv_dataset(input_path):
    tasks = [
        "MUV-466",
        "MUV-548",
        "MUV-600",
        "MUV-644",
        "MUV-652",
        "MUV-689",
        "MUV-692",
        "MUV-712",
        "MUV-713",
        "MUV-733",
        "MUV-737",
        "MUV-810",
        "MUV-832",
        "MUV-846",
        "MUV-852",
        "MUV-858",
        "MUV-859",
    ]
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df[tasks]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    # convert 0 to -1, then nan to 0
    # so MUV has three values, -1, 0, 1
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values)


def _load_sider_dataset(input_path):
    tasks = [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ]

    input_df = pd.read_csv(input_path, sep=",")
    smiles_list, labels = input_df["smiles"], input_df[tasks]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return (smiles_list, rdkit_mol_objs_list, labels.values)


def _load_toxcast_dataset(input_path):
    # Note: some examples have multiple species,
    #   some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [
        m if m is not None else None for m in rdkit_mol_objs_list
    ]
    preprocessed_smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None
        for m in preprocessed_rdkit_mol_objs_list
    ]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1, then nan to 0
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)

    non_loc = [i for (i, v) in enumerate(rdkit_mol_objs_list) if v is not None]
    sel_non = lambda a_list: [a_list[index] for index in non_loc]

    return (
        sel_non(preprocessed_smiles_list),
        sel_non(preprocessed_rdkit_mol_objs_list),
        labels.values[non_loc],
    )


def _load_chembl_with_labels_dataset(root_path):
    """
    Large-scale comparison of MLs methods for drug target prediction on ChEMBL
    :param root_path: folder that contains the reduced chembl dataset
    :return:
        list of smiles,
        preprocessed rdkit mol obj list,
        list of np.array containing indices for each of the 3 folds,
        np.array containing the labels"""
    # ref https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/
    #   chembl20/dataPythonReduced/chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/
    #   chembl20/dataPythonReduced/chembl20LSTM.pckl

    """ 1. load folds and labels """
    f = open(join(root_path, "folds0.pckl"), "rb")
    folds = pickle.load(f)
    f.close()

    f = open(join(root_path, "labelsHard.pckl"), "rb")
    # targetAnnInd = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetMat = pickle.load(f)
    f.close()

    # targetMat = targetMat
    targetMat = targetMat.copy().tocsr()
    targetMat.sort_indices()
    # targetAnnInd = targetAnnInd
    # targetAnnInd = targetAnnInd - targetAnnInd.min()

    folds = [np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed = targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    # trainPosOverall = np.array(
    #     [
    #         np.sum(targetMatTransposed[x].data > 0.5)
    #         for x in range(targetMatTransposed.shape[0])
    #     ]
    # )
    # # num negative examples in each of the 1310 targets
    # trainNegOverall = np.array(
    #     [
    #         np.sum(targetMatTransposed[x].data < -0.5)
    #         for x in range(targetMatTransposed.shape[0])
    #     ]
    # )
    # dense array containing the labels for 456331 molecules and 1310 targets
    denseOutputData = targetMat.A  # possible values are {-1, 0, 1}

    """ 2. load structures """
    f = open(join(root_path, "chembl20LSTM.pckl"), "rb")
    rdkitArr = pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print("preprocessing")
    for i in range(len(rdkitArr)):
        print(i)
        m = rdkitArr[i]
        if m is None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [
        AllChem.MolToSmiles(m) if m is not None else None for m in preprocessed_rdkitArr
    ]
    # bc some empty mol in the rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    non_loc = [i for (i, v) in enumerate(smiles_list) if v is not None]
    sel_non = lambda a_list: [a_list[index] for index in non_loc]

    # todo: folds might to be changed
    return (
        sel_non(smiles_list),
        sel_non(preprocessed_rdkitArr),
        folds,
        denseOutputData[non_loc].reshape(-1, 1),
    )


class GEOMDataset(InMemoryDataset):
    """Pre-Training 2D/3D GNN with GEOM"""

    def __init__(
        self,
        root,
        n_mol,
        n_conf,
        n_upper=9999,
        seed=777,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        empty=False,
        **kwargs
    ):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, "raw"), exist_ok=True)
        os.makedirs(join(root, "processed"), exist_ok=True)

        if "smiles_copy_from_3D_file" in kwargs:  # for 2D Datasets (SMILES)
            self.smiles_copy_from_3D_file = kwargs["smiles_copy_from_3D_file"]
        else:
            self.smiles_copy_from_3D_file = None

        self.root, self.seed = root, seed
        self.n_mol, self.n_conf, self.n_upper = n_mol, n_conf, n_upper
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(GEOMDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print(
            "root: {},\ndata: {},\nn_mol: {},\nn_conf: {}".format(
                self.root, self.data, self.n_mol, self.n_conf
            )
        )

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        return

    def process(self):
        data_list = []
        smiles_list = []

        if self.smiles_copy_from_3D_file is None:  # 3D datasets
            dir_name = join(GEOM_ROOT, "rdkit_folder")
            drugs_file = "{}/summary_drugs.json".format(dir_name)
            with open(drugs_file, "r") as f:
                drugs_summary = json.load(f)
            drugs_summary = list(drugs_summary.items())
            print("# SMILES: {}".format(len(drugs_summary)))
            # expected: 304,466 molecules

            random.seed(self.seed)
            random.shuffle(drugs_summary)
            mol_idx, idx, notfound = 0, 0, 0
            for smiles, sub_dic in tqdm(drugs_summary):

                """path should match"""
                if sub_dic.get("pickle_path", "") == "":
                    notfound += 1
                    continue

                mol_path = join(dir_name, sub_dic["pickle_path"])
                with open(mol_path, "rb") as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic["conformers"]

                    """ energy should ascend, which turns out it doesn't"""
                    energy_list = [item["relativeenergy"] for item in conformer_list]
                    # assert np.all(np.diff(energy_list) >= 0)
                    conformer_list = [
                        conformer_list[i] for i in np.argsort(energy_list)
                    ]

                    """ count should match """
                    # there are other ways (e.g. repeated sampling) for molecules that do not have enough conformers
                    conf_n = len(conformer_list)
                    # if conf_n < self.n_conf or conf_n > self.n_upper:
                    if conf_n < self.n_conf:
                        notfound += 1
                        continue

                    """ SMILES should match """
                    # Ref:
                    # https://github.com/learningmatter-mit/geom/issues/4#issuecomment-853486681
                    # https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb
                    conf_list = [
                        Chem.MolToSmiles(
                            Chem.MolFromSmiles(Chem.MolToSmiles(rd_mol["rd_mol"]))
                        )
                        for rd_mol in conformer_list[: self.n_conf]
                    ]

                    conf_list_raw = [
                        Chem.MolToSmiles(rd_mol["rd_mol"])
                        for rd_mol in conformer_list[: self.n_conf]
                    ]
                    # check that they're all the same
                    same_confs = len(list(set(conf_list))) == 1
                    same_confs_raw = len(list(set(conf_list_raw))) == 1
                    if not same_confs:
                        # print(list(set(conf_list)))
                        if same_confs_raw is True:
                            print("Interesting")
                        notfound += 1
                        continue

                    for conformer_dict in conformer_list[: self.n_conf]:
                        rdkit_mol = conformer_dict["rd_mol"]
                        data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                        data.mol_id = torch.tensor([mol_idx])
                        data.id = torch.tensor([idx])
                        smiles_list.append(smiles)
                        data_list.append(data)
                        idx += 1

                if mol_idx + 1 >= self.n_mol:
                    break
                if same_confs:
                    mol_idx += 1

            print(
                "mol id: [0, {}]\tsmiles: {}\tset(smiles): {}".format(
                    mol_idx, len(smiles_list), len(set(smiles_list))
                )
            )

        else:  # 2D datasets
            with open(self.smiles_copy_from_3D_file, "r") as f:
                lines = f.readlines()
            for smiles in lines:
                smiles_list.append(smiles.strip())
            smiles_list = list(dict.fromkeys(smiles_list))

            # load 3D structure
            dir_name = join(GEOM_ROOT, "rdkit_folder")
            drugs_file = "{}/summary_drugs.json".format(dir_name)
            with open(drugs_file, "r") as f:
                drugs_summary = json.load(f)
            # expected: 304,466 molecules
            print("number of SMILES: {}".format(len(drugs_summary.items())))

            mol_idx, idx, notfound = 0, 0, 0

            for smiles in tqdm(smiles_list):
                sub_dic = drugs_summary[smiles]
                mol_path = join(dir_name, sub_dic["pickle_path"])
                with open(mol_path, "rb") as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic["conformers"]
                    conformer = conformer_list[0]
                    rdkit_mol = conformer["rd_mol"]
                    data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                    data.mol_id = torch.tensor([mol_idx])
                    data.id = torch.tensor([idx])
                    data_list.append(data)
                    mol_idx += 1
                    idx += 1

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        smiles_series = pd.Series(smiles_list)
        saver_path = join(self.processed_dir, "smiles.csv")
        print("saving to {}".format(saver_path))
        smiles_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("%d molecules do not meet the requirements" % notfound)
        print("%d molecules have been processed" % mol_idx)
        print("%d conformers have been processed" % idx)
        return


""" unused utilities in the reference code """
# def merge_dataset_objs(dataset_1, dataset_2):
# def create_circular_fingerprint(mol, radius, size, chirality):
# class MoleculeFingerprintDataset(data.Dataset):


if __name__ == "__main__":

    """=== Process downstream datasets ==="""
    # regression datasets: esol, freesolv, lipophilicity,
    # classification datasets that have [-1, 0, 1]: muv, tox21, toxcast, others are binary
    downstream_dir = [
        "bace",
        "bbbp",
        "clintox",
        "esol",
        "freesolv",
        "hiv",
        "lipophilicity",
        "muv",
        "sider",
        "tox21",
        "toxcast",
        # "chembl_filtered",
        # "zinc_standard_agent",
    ]

    """ Pre-Training, 2M ZINC15"""
    downstream_dir = [
        "zinc_standard_agent",
    ]

    for dataset in downstream_dir:
        root = join(DOWNSTREAM_ROOT, dataset, "processed")
        shutil.rmtree(root) if REMOVE_CACHE else None
        os.makedirs(root, exist_ok=True)
        MoleculeDataset(join(DOWNSTREAM_ROOT, dataset), dataset=dataset)
        print("=" * 27)
        print("\n")

    """ === Process GEOM datasets === """
    # for n_conf in [5, 10]:
    #     # for n_conf in [1]:  # [1, 5, 10, 20]
    #     for n_mol in [500000]:  # [50000, 100000, 200000, 500000]:
    #         root_2d = join(GEOM_ROOT, "geom2d_nmol%d_nconf%d/" % (n_mol, n_conf))
    #         root_3d = join(GEOM_ROOT, "geom3d_nmol%d_nconf%d/" % (n_mol, n_conf))
    #         if REMOVE_CACHE:
    #             shutil.rmtree(root_2d) if os.path.exists(root_2d) else None
    #             shutil.rmtree(root_3d) if os.path.exists(root_3d) else None

    #         GEOMDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf)
    #         GEOMDataset(
    #             root=root_2d,
    #             n_mol=n_mol,
    #             n_conf=n_conf,
    #             smiles_copy_from_3D_file="%s/processed/smiles.csv" % root_3d,
    #         )
