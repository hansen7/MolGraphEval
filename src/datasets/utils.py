import torch, pickle, numpy as np, networkx as nx
from torch_geometric.data import Data
from multiprocessing import Pool
from collections import Counter
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm

# allowable node and edge features, used in molecule_datasets.py
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}
feats = allowable_features


def mol_to_graph_data_obj_simple(mol):
    """used in MoleculeDataset() class in molecular_datasets.py
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr"""

    """ atom/node """
    # num_atom_features = 2  # atom type, chirality tag
    atom_feats_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            feats["possible_atomic_num_list"].index(atom.GetAtomicNum())
        ] + [feats["possible_chirality_list"].index(atom.GetChiralTag())]
        atom_feats_list.append(atom_feature)
    x = torch.tensor(np.array(atom_feats_list), dtype=torch.long)

    """ bond/edge """
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [feats["possible_bonds"].index(bond.GetBondType())] + [
                feats["possible_bond_dirs"].index(bond.GetBondDir())
            ]
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the PyG.
    Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr"""
    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_feats_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            feats["possible_atomic_num_list"].index(atom.GetAtomicNum())
        ] + [feats["possible_chirality_list"].index(atom.GetChiralTag())]
        atom_feats_list.append(atom_feature)
    x = torch.tensor(np.array(atom_feats_list), dtype=torch.long)

    # bonds
    num_bond_feats = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [feats["possible_bonds"].index(bond.GetBondType())] + [
                feats["possible_bond_dirs"].index(bond.GetBondDir())
            ]
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_feats), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    z = []
    for atom in mol.GetAtoms():
        z.append(atom.GetAtomicNum())
    z = torch.Tensor(np.array(z)).long()

    return Data(
        x=x, z=z, edge_index=edge_index, edge_attr=edge_attr, positions=positions
    )


def graph_data_obj_to_mol_simple(data_x, data_edge_index, data_edge_attr):
    """Inverse of mol_to_graph_data_obj_simple()"""
    mol = Chem.RWMol()

    # atoms
    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        atomic_num = feats["possible_atomic_num_list"][atomic_num_idx]
        chirality_tag = feats["possible_chirality_list"][chirality_tag_idx]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        mol.AddAtom(atom)

    # bonds
    edge_index = data_edge_index.cpu().numpy()
    edge_attr = data_edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        bond_type = feats["possible_bonds"][bond_type_idx]
        bond_dir = feats["possible_bond_dirs"][bond_dir_idx]
        mol.AddBond(begin_idx, end_idx, bond_type)
        # set bond direction
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        new_bond.SetBondDir(bond_dir)

    return mol


def graph_data_obj_to_nx_simple(data):
    """torch geometric -> networkx
    NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: networkx object"""
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        G.add_node(
            node_for_adding=i,
            atom_num_idx=atomic_num_idx,
            chirality_tag_idx=chirality_tag_idx,
        )
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(
                u_of_edge=begin_idx,
                v_of_edge=end_idx,
                bond_type_idx=bond_type_idx,
                bond_dir_idx=bond_dir_idx,
            )
    return G


def nx_to_graph_data_obj_simple(G):
    """vice versa of graph_data_obj_to_nx_simple()
    Assume node indices are numbered from 0 to num_nodes - 1.
    NB: Uses simplified atom and bond features, and represent as indices.
    NB: possible issues with recapitulating relative stereochemistry
        since the edges in the nx object are unordered."""

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node["atom_num_idx"], node["chirality_tag_idx"]]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge["bond_type_idx"], edge["bond_dir_idx"]]
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        return False
    except:
        return False


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one."""

    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def create_standardized_mol_id(smiles):
    """smiles -> inchi"""

    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(
            AllChem.MolFromSmiles(smiles), isomericSmiles=False
        )
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            if "." in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
    return


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively."""

    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split(".")
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


# def get_gasteiger_partial_charges(mol, n_iter=12):
#     """
#     Calculates list of gasteiger partial charges for each atom in mol object.
#     :param mol: rdkit mol object
#     :param n_iter: number of iterations Default 12
#     :return: list of computed partial charges for each atom. """
#
#     # Chem.rdPartialCharges.ComputeGasteigerCharges(
#     #     mol, nIter=n_iter, throwOnParamFailure=True)
#     Chem.rdPartialCharges(
#         mol, nIter=n_iter, throwOnParamFailure=True)
#     partial_charges = [float(a.GetProp('_GasteigerCharge'))
#                        for a in mol.GetAtoms()]
#     return partial_charges


def atom_to_vocab(mol, atom):
    """
    Convert atom to vocabulary, based on atom and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated atom vocabulary with its contexts."""
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])

    # The generated atom_vocab is too long?
    return output


BOND_FEATURES = ["BondType", "BondDir"]


def get_bond_feature_name(bond):
    """
    Return the string format of bond features.
    Bond features are surrounded with ()"""
    ret = []
    for bond_feature in BOND_FEATURES:
        fea = eval(f"bond.Get{bond_feature}")()
        ret.append(str(fea))

    return "(" + "-".join(ret) + ")"


def bond_to_vocab(mol, bond):
    """
    Convert bond to vocabulary, based on atom and bond type.
    Considering one-hop neighbor atoms
    :param mol: the molecular.
    :return: the generated bond vocabulary with its contexts."""
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + "-" + get_bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = get_bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output


class TorchVocab(object):
    def __init__(
        self,
        counter,
        max_size=None,
        min_freq=1,
        specials=("<pad>", "<other>"),
        vocab_type="atom",
    ):
        """
        :param counter:
        :param max_size:
        :param min_freq:
        :param specials:
        :param vocab_type: 'atom': atom atom_vocab; 'bond': bond atom_vocab."""
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        if vocab_type in ("atom", "bond"):
            self.vocab_type = vocab_type
        else:
            raise ValueError("Wrong input for vocab_type!")
        self.itos = list(specials)

        max_size = None if max_size is None else max_size + len(self.itos)
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        # stoi is simply a reversed dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.other_index = 1
        self.pad_index = 0

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        # if self.vectors != other.vectors:
        #    return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
                self.freqs[w] = 0
            self.freqs[w] += v.freqs[w]

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class MolVocab(TorchVocab):
    def __init__(
        self,
        molecule_list,
        max_size=None,
        min_freq=1,
        num_workers=1,
        total_lines=None,
        vocab_type="atom",
    ):
        if vocab_type in ("atom", "bond"):
            self.vocab_type = vocab_type
        else:
            raise ValueError("Wrong input for vocab_type!")
        print("Building {} vocab from mol-list".format(self.vocab_type))

        from rdkit import RDLogger

        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        if total_lines is None:
            total_lines = len(molecule_list)

        res = []
        batch = 50000
        counter = Counter()
        pool = Pool(num_workers)
        pbar = tqdm(total=total_lines)
        callback = lambda a: pbar.update(batch)
        for i in range(int(total_lines / batch + 1)):
            start = int(batch * i)
            end = min(total_lines, batch * (i + 1))
            res.append(
                pool.apply_async(
                    MolVocab.read_counter_from_molecules,
                    args=(
                        molecule_list,
                        start,
                        end,
                        vocab_type,
                    ),
                    callback=callback,
                )
            )
        pool.close()
        pool.join()
        for r in res:
            sub_counter = r.get()
            for k in sub_counter:
                if k not in counter:
                    counter[k] = 0
                counter[k] += sub_counter[k]
        super().__init__(
            counter, max_size=max_size, min_freq=min_freq, vocab_type=vocab_type
        )

    @staticmethod
    def read_counter_from_molecules(molecule_list, start, end, vocab_type):
        sub_counter = Counter()
        for i, mol in enumerate(molecule_list):
            if i < start:
                continue
            if i >= end:
                break
            if vocab_type == "atom":
                for atom in mol.GetAtoms():
                    v = atom_to_vocab(mol, atom)
                    sub_counter[v] += 1
            else:
                for bond in mol.GetBonds():
                    v = bond_to_vocab(mol, bond)
                    sub_counter[v] += 1
        return sub_counter

    @staticmethod
    def load_vocab(vocab_path: str) -> "MolVocab":
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
