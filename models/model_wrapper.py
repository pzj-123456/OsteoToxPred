import json
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
from models.ginet_finetune import GINet
from dataset.dataset_test import MolTestDatasetWrapper
from models.attention_visualizer import AttentionVisualizer

class ToxicityPredictor:
    """毒性预测模型包装器，用于Flask应用"""
    
    def __init__(self, config_path="config_flask.yaml"):
        self.config = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
        self.device = self._get_device()
        self.model = None
        self.normalizer = None
        self.attention_visualizer = AttentionVisualizer()
        self._load_model()
    
    def _get_device(self):
        """获取设备配置"""
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            # 配置数据集
            self.config['dataset']['task'] = 'classification'
            self.config['dataset']['data_path'] = 'data/test.csv'
            self.config['dataset']['target'] = 'label'
            
            # 创建模型
            fps = self.config['dataset']['fingerprint_list']
            self.model = GINet(
                self.config['dataset']['task'],
                fingerprint_list=fps, 
                **self.config["model"]
            ).to(self.device)
            
            # 加载预训练权重
            model_path = os.path.join("./ckpt", 'checkpoints', 'model.pth')
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("模型加载成功")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def predict_batch(self, smiles_list):
        """批量预测SMILES的毒性"""
        try:
            # 创建临时配置文件
            temp_config = self.config.copy()
            temp_config['dataset']['data_path'] = 'temp_data.csv'
            
            # 创建临时数据文件
            temp_df = pd.DataFrame({'smiles': smiles_list})
            temp_df.to_csv('temp_data.csv', index=False)
            
            # 创建数据集
            dataset = MolTestDatasetWrapper(
                batch_size=self.config['batch_size'],
                num_workers=temp_config['dataset']['num_workers'],
                valid_size=temp_config['dataset']['valid_size'],
                test_size=temp_config['dataset']['test_size'],
                data_path=temp_config['dataset']['data_path'],
                target=temp_config['dataset']['target'],
                task=temp_config['dataset']['task'],
                splitting=temp_config['dataset']['splitting'],
                fingerprint_list=temp_config['dataset']['fingerprint_list'],
                fp_radius=temp_config['dataset']['fp_radius'],
                ecfp_bits=temp_config['dataset']['ecfp_bits'],
                maccs_bits=temp_config['dataset']['maccs_bits'],
                ap_bits=temp_config['dataset']['ap_bits'],
                ext_bits=temp_config['dataset']['ext_bits'],
                extfp_maxPath=temp_config['dataset']['extfp_maxPath'],
                torsion_bits=temp_config['dataset']['torsion_bits'],
                avalon_bits=temp_config['dataset']['avalon_bits']
            )
            test_dataset, test_loader = dataset.get_data_loaders()
            
            # 预测
            results = []
            all_smiles = []
            all_preds = []
            all_attns = []
            
            with torch.no_grad():
                self.model.eval()
                
                for bn, data in enumerate(test_loader):
                    data = data.to(self.device)
                    
                    __, pred, node_attn = self.model(data)
                    
                    if self.normalizer:
                        pred = self.normalizer.denorm(pred)
                    
                    # 分类任务使用softmax
                    if self.config['dataset']['task'] == 'classification':
                        pred = F.softmax(pred, dim=-1)
                    
                    # 收集预测结果和注意力
                    smiles_batch = data.z
                    pred_scores = pred[:, 1] if self.config['dataset']['task'] == 'classification' else pred.flatten()
                    pred_vals = pred_scores.cpu().tolist()
                    
                    node_attn = node_attn.cpu().detach().numpy()
                    batch_idx = data.batch.cpu().numpy()
                    
                    for i, smi in enumerate(smiles_batch):
                        mask = (batch_idx == i)
                        attn_per_graph = node_attn[mask].tolist()
                        
                        all_smiles.append(smi)
                        all_preds.append(pred_vals[i])
                        all_attns.append(attn_per_graph)
                        
                        # 构建结果
                        result = {
                            'smiles': smi,
                            'toxicity_score': float(pred_vals[i]),
                            'toxicity_probability': float(pred_vals[i]),
                            'attention_weights': attn_per_graph
                        }
                        results.append(result)
            
            # 清理临时文件
            if os.path.exists('temp_data.csv'):
                os.remove('temp_data.csv')
            
            return results
            
        except Exception as e:
            print(f"预测失败: {e}")
            raise
    
    def predict_single(self, smiles):
        """预测单个SMILES的毒性"""
        results = self.predict_batch([smiles])
        return results[0] if results else None
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'GINet',
            'task': self.config['dataset']['task'],
            'device': str(self.device),
            'config': self.config
        }
    
    def create_attention_visualization(self, smiles, attention_weights, job_id, idx):
        """创建注意力权重可视化"""
        try:
            return self.attention_visualizer.create_combined_visualization(
                smiles, attention_weights, job_id, idx
            )
        except Exception as e:
            print(f"创建注意力可视化失败: {e}")
            return {
                'molecule_image': "/static/img/default_molecule.png",
                'heatmap_image': "/static/img/default_molecule.png",
                'attention_stats': {}
            }
