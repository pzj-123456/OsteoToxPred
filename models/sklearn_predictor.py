import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, RDKFingerprint
from rdkit.Avalon import pyAvalonTools
import joblib

MODEL_DIR = Path(__file__).resolve().parent / "sklearn_models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

FINGERPRINT_FUNCS = [
    ("Morgan", lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), 2048),
    ("MACCS", lambda mol: MACCSkeys.GenMACCSKeys(mol), 166),
    ("AtomPair", lambda mol: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048), 2048),
    ("RDK", lambda mol: RDKFingerprint(mol, maxPath=5, fpSize=2048), 2048),
    ("TopologicalTorsion", lambda mol: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048), 2048),
    ("Avalon", lambda mol: pyAvalonTools.GetAvalonFP(mol, nBits=512), 512),
]

class SklearnPredictor:
    """单模型+六指纹拼接"""

    def __init__(self, model_name: str):
        print(f"DEBUG: 加载模型 {model_name}")
        model_file_candidates = [f"{model_name}.pkl", f"{model_name}_model.pkl"]
        model_path = None
        for fname in model_file_candidates:
            path = MODEL_DIR / fname
            if path.exists():
                model_path = path; break
        if model_path is None:
            raise FileNotFoundError(f"未找到模型文件: {model_file_candidates} 在 {MODEL_DIR}")
        
        print(f"DEBUG: 使用模型文件: {model_path}")
        try:
            self.clf = joblib.load(model_path)
            print(f"DEBUG: 使用joblib加载成功")
        except Exception as e:
            print(f"DEBUG: joblib加载失败: {e}, 尝试pickle")
            with open(model_path, "rb") as f:
                self.clf = pickle.load(f)
            print(f"DEBUG: 使用pickle加载成功")
        
        print(f"DEBUG: 模型类型: {type(self.clf)}")
        # 计算总bits
        self.total_bits = sum(bits for _,_,bits in FINGERPRINT_FUNCS)
        print(f"DEBUG: 总特征维度: {self.total_bits}")

    def _smiles_to_bits(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        concat = []
        for _, func, bits in FINGERPRINT_FUNCS:
            fp = func(mol)
            if fp is None:
                return None
            arr = np.zeros((bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            concat.append(arr)
        return np.concatenate(concat)

    def predict_batch(self, smiles_list: List[str]):
        print(f"DEBUG: 开始预测 {len(smiles_list)} 个分子")
        features: List[np.ndarray] = []
        valid_idx: List[int] = []
        for idx, smi in enumerate(smiles_list):
            arr = self._smiles_to_bits(smi)
            if arr is not None:
                features.append(arr)
                valid_idx.append(idx)
            else:
                print(f"DEBUG: SMILES {smi} 无法转换为指纹")
        if not features:
            raise ValueError("没有有效的SMILES可用于预测")
        X = np.array(features)
        print(f"DEBUG: 特征矩阵形状: {X.shape}")
        print(f"DEBUG: 特征矩阵数据类型: {X.dtype}")
        print(f"DEBUG: 特征矩阵前5个值: {X[0][:5] if len(X) > 0 else 'empty'}")
        
        # 确保模型是确定性的
        if hasattr(self.clf, 'random_state'):
            print(f"DEBUG: 模型随机状态: {self.clf.random_state}")
        
        probs = self.clf.predict_proba(X)[:, 1]
        print(f"DEBUG: 预测概率: {probs}")
        
        results = []
        j = 0
        for i in range(len(smiles_list)):
            if i in valid_idx:
                p = float(probs[j]); j += 1
            else:
                p = 0.5  # 默认概率
            results.append({
                "toxicity_probability": p,
                "attention_weights": []
            })
        return results
