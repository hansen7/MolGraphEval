import json


def clean_rdkit_file(data):
    """Write smiles in a csv file"""

    dir_name = "rdkit_folder"
    drugs_file = "{}/summary_{}.json".format(dir_name, data)
    with open(drugs_file, "r") as f:
        drugs_summary = json.load(f)

    # 304,466 molecules in total
    smiles_list = list(drugs_summary.keys())
    print("Number of total items (SMILES): {}".format(len(smiles_list)))

    drug_file = "{}.csv".format(data)
    with open(drug_file, "w") as f:
        f.write("smiles\n")
        for smiles in smiles_list:
            f.write("{}\n".format(smiles))

    return


if __name__ == "__main__":
    data = "drugs"
    clean_rdkit_file(data)
