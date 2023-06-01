#  Copyright (c) 2021. Shengchao & Hanchen
#  liusheng@mila.quebec & hw501@cam.ac.uk

import hashlib
import json
import os
import pickle
import random
from os.path import join

import msgpack
from rdkit import Chem

# from rdkit.Chem import AllChem
# from rdkit.Chem.Draw import MolsToGridImage
# from rdkit.Chem.rdmolfiles import MolToPDBFile
# from rdkit.Chem.rdMolDescriptors import CalcWHIM

zip2md5 = {
    "drugs_crude.msgpack.tar.gz": "7778e84c50b7cde755cca670d1f75091",
    "drugs_featurized.msgpack.tar.gz": "2fb86edc50e3ab3b96f78fe01082965b",
    "qm9_crude.msgpack.tar.gz": "aad0081ed5d9b8c93c2bd0235987573b",
    "qm9_featurized.msgpack.tar.gz": "09655f470f438e3a7a0dfd20f40f6f22",
    "rdkit_folder.tar.gz": "e8f2168b7050652db22c976be25c450e",
}


def compute_md5(file_name, chunk_size=65536):
    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()


def analyze_crude_file(data):
    drugs_file = "{}_crude.msgpack".format(data)
    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))
    print(compute_md5("{}.tar.gz".format(drugs_file)))
    print(zip2md5["{}.tar.gz".format(drugs_file)])

    total_smiles_list = []
    for idx, drug_batch in enumerate(unpacker):
        smiles_list = list(drug_batch.keys())
        print(idx, "\t", len(smiles_list))
        total_smiles_list.extend(smiles_list)

        for smiles in smiles_list:
            print(smiles)
            if smiles == "CCOCC[C@@H](O)C=O":
                print(drug_batch[smiles])
                conformer_list = drug_batch[smiles]["conformers"]
                print(len(conformer_list))
            # break
        break
    print("total smiles list {}".format(len(total_smiles_list)))
    return


def analyze_featurized_file(data):
    drugs_file = "{}_featurized.msgpack".format(data)
    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))
    print(compute_md5("{}.tar.gz".format(drugs_file)))
    print(zip2md5["{}.tar.gz".format(drugs_file)])

    for idx, drug_batch in enumerate(unpacker):
        smiles_list = list(drug_batch.keys())
        for smiles in smiles_list:
            print(smiles)
            print(len(drug_batch[smiles]))
            # print(drug_batch[smiles])
            break
        break

    return


def analyze_rdkit_file(data):
    dir_name = "rdkit_folder"
    # dir_zip_name = '{}.tar.gz'.format(dir_name)
    # assert compute_md5(dir_zip_name) == zip2md5[dir_zip_name]

    drugs_file = "{}/summary_{}.json".format(dir_name, data)
    with open(drugs_file, "r") as f:
        drugs_summary = json.load(f)

    smiles_list = list(drugs_summary.keys())
    print("# SMILES: {}".format(len(smiles_list)))  # 304,466
    example_smiles = smiles_list[0]
    print(drugs_summary[example_smiles])

    # Now let's find active molecules and their pickle paths:
    active_mol_paths = []
    active_smiles_list = []
    for smiles, sub_dic in drugs_summary.items():
        if sub_dic.get("sars_cov_one_cl_protease_active") == 1:
            pickle_path = join(dir_name, sub_dic.get("pickle_path", ""))
            if os.path.isfile(pickle_path):
                active_mol_paths.append(pickle_path)
    print("# active mols on CoV 3CL: {}\n".format(len(active_mol_paths)))

    # Now randomly sample inactive molecules and their pickle paths:
    random_smiles = list(drugs_summary.keys())
    random.shuffle(random_smiles)
    random_smiles = random_smiles[:1000]
    inactive_mol_paths = []
    for smiles in random_smiles:
        sub_dic = drugs_summary[smiles]
        if sub_dic.get("sars_cov_one_cl_protease_active") == 0:
            pickle_path = join(dir_name, sub_dic.get("pickle_path", ""))
            if os.path.isfile(pickle_path):
                inactive_mol_paths.append(pickle_path)
    print("# inactive mols on CoV 3CL: {}\n".format(len(inactive_mol_paths)))

    sample_dic = {}
    sample_smiles = active_smiles_list
    sample_smiles.extend(random_smiles)
    for mol_path in [*active_mol_paths, *inactive_mol_paths]:
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        sample_dic.update({dic["smiles"]: dic})
    print("# all mols on CoV 3CL: {}\n".format(len(sample_dic)))

    idx = 0
    for k, v in sample_dic.items():
        conf_list = v["conformers"]
#         print(k)
#         print(len(conf_list))
        for conf in conf_list:
            mol = conf["rd_mol"]
#             print(conf)
#             print(mol)
            print(Chem.MolToSmiles(mol))
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
            print(smiles)
        print("\n")

        idx += 1
        if idx > 9:
            break

    return


if __name__ == "__main__":
    data = "drugs"
    # analyze_crude_file(data)
    # analyze_featurized_file(data)
    analyze_rdkit_file(data)