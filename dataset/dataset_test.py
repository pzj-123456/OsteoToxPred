import os
import csv
import math
import time
import random
from typing import List
from collections import defaultdict
import numpy as np

import torch
import torch.nn.functional as F
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.utils import shuffle
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, rdMolDescriptors, Fingerprints
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger

# 下面导入 RDKit 相关包，用于生成各种指纹
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import RDKFingerprint

RDLogger.DisableLog('rdApp.*')  

# 定义可用的指纹名称列表
# 顶部新增
AVAILABLE_FPS = ['ecfp','maccs','ap','ext','torsion','avalon']

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain assert from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def random_scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    index = np.array(list(range(0, len(dataset))))
    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(dataset.smiles_data):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(valid_size * len(dataset)))
    n_total_test = int(np.floor(test_size * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_index, val_index, test_index = index[train_idx], index[valid_idx], index[test_idx]

    # if sort:
    #     train_index = sorted(train_index)
    #     val_index = sorted(val_index)
    #     test_index = sorted(test_index)

    return train_index, val_index, test_index


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    # data_path = '..\data\test.csv'
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row['smiles']
            # label = row[target]
            mol = Chem.MolFromSmiles(smiles)
            if mol != None :
                smiles_data.append(smiles)         
            else:
                print("mol is None")
                ValueError('task must be either regression or classification')
    # smiles_data, labels = shuffle(smiles_data, labels, random_state=42)
    print(f"data_path：{data_path}")

    print(f"smiles_data：{smiles_data}")

    print(f"Total samples: {len(smiles_data)} (shuffled with seed=42)")
    return smiles_data


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task,
                 fingerprint_list=None,
                 fp_radius=2,
                 ecfp_bits=2048,
                 maccs_bits=167,
                 ap_bits=2048,
                 ext_bits=2048,
                 torsion_bits=2048,
                 avalon_bits=1024,

                 extfp_maxPath=5):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path, target, task)
        self.task = task

        # 用户选择的指纹列表
        self.fingerprint_list = fingerprint_list or AVAILABLE_FPS
        # 验证用户输入
        for fp_name in self.fingerprint_list:
            if fp_name not in AVAILABLE_FPS:
                raise ValueError(f"Unsupported fingerprint: {fp_name}")

        # ---------------- 指纹相关的超参数 ----------------
        self.fp_radius = fp_radius        # ECFP 半径
        self.ecfp_bits = ecfp_bits        # ECFP 位长
        self.maccs_bits = maccs_bits      # MACCS 位长（通常 167）
        self.atompair_bits = ap_bits  # AtomPair 位长
        self.extfp_bits = ext_bits      # ExtFP 位长
        self.extfp_maxPath = extfp_maxPath  # ExtFP 最大路径长度

        self.torsion_bits = torsion_bits
        self.avalon_bits = avalon_bits



        # ------------------------------------------------


        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        z = self.smiles_data[index]
        mol = Chem.MolFromSmiles(z)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
 

        # ----------------- 新增：计算四种指纹并拼接 -----------------

        # ================= 计算所需指纹 =================
        arrs = []
        if 'ecfp' in self.fingerprint_list:
            fp_ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.fp_radius, nBits=self.ecfp_bits)
            arrs.append(np.array(fp_ecfp, dtype=np.float32))
        if 'maccs' in self.fingerprint_list:
            fp_maccs = MACCSkeys.GenMACCSKeys(mol)
            arrs.append(np.array(fp_maccs, dtype=np.float32))
        if 'ap' in self.fingerprint_list:
            fp_ap = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=self.atompair_bits)
            arrs.append(np.array(fp_ap, dtype=np.float32))
        if 'ext' in self.fingerprint_list:
            fp_ext = RDKFingerprint(mol, maxPath=self.extfp_maxPath, fpSize=self.extfp_bits)
            arrs.append(np.array(fp_ext, dtype=np.float32))

        if 'torsion' in self.fingerprint_list:
            fp_torsion = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=self.torsion_bits
            )
            arrs.append(np.array(fp_torsion, dtype=np.float32))


        if 'avalon' in self.fingerprint_list:
            # 使用正确的 Avalon FP 计算方法
            fp_avalon = GetAvalonFP(mol, nBits=self.avalon_bits)
            arrs.append(np.array(fp_avalon, dtype=np.float32))



        # 拼接所有选中指纹
        arr_concat = np.concatenate(arrs, axis=0)
        fp_tensor = torch.from_numpy(arr_concat)
        # ================================================

        # 转为 Torch Tensor
        fp_tensor = torch.from_numpy(arr_concat)  # shape: [总指纹维度]

        data = Data(x=x, z=z, edge_index=edge_index, edge_attr=edge_attr)

        data.fp = fp_tensor  # 新增一个属性：data.fp 形状是 [总指纹维度]
        # print(data.fp.shape)
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting,fingerprint_list=None,
                 fp_radius=2,
                 ecfp_bits=2048,
                 maccs_bits=167,
                 ap_bits=2048,
                 ext_bits=2048,
                 extfp_maxPath=5,
                 torsion_bits=2048,
                 avalon_bits=1024
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        assert splitting in ['random', 'scaffold', 'random_scaffold']
        self.fingerprint_list = fingerprint_list or AVAILABLE_FPS

        self.fp_radius = fp_radius
        self.ecfp_bits = ecfp_bits
        self.maccs_bits = maccs_bits
        self.ap_bits = ap_bits
        self.ext_bits = ext_bits
        self.extfp_maxPath = extfp_maxPath
        self.torsion_bits = torsion_bits
        self.avalon_bits = avalon_bits

    def get_data_loaders(self):
        # 创建包含全部数据的测试数据集
        test_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task,
                                       fingerprint_list=self.fingerprint_list,
                                       fp_radius=self.fp_radius,
                                       ecfp_bits=self.ecfp_bits,
                                       maccs_bits=self.maccs_bits,
                                       ap_bits=self.ap_bits,
                                       ext_bits=self.ext_bits,
                                       extfp_maxPath=self.extfp_maxPath,
                                       torsion_bits=self.torsion_bits,
                                       avalon_bits=self.avalon_bits
                                       )
        
        # 创建测试数据加载器，使用全部数据
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=False
        )
        
        return test_dataset, test_loader
    


    def get_train_validation_data_loaders(self, train_dataset,seed=42):
        random.seed(seed)
        if self.splitting == 'random':
            # 设置固定的随机种子
            np.random.seed(42)  # 设置随机种子为 42，确保每次划分一致

            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size,seed)
        elif self.splitting == 'random_scaffold':
            train_idx, valid_idx, test_idx = random_scaffold_split(train_dataset, self.valid_size, self.test_size, seed)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=True
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=True
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=True
        )

        return train_loader, valid_loader, test_loader
