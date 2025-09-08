#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力权重可视化模块
用于将注意力权重映射到2D分子结构图上
"""

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import uuid

class AttentionVisualizer:
    """注意力权重可视化器"""
    
    def __init__(self, output_dir="static/tmp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def normalize_attention_weights(self, attention_weights):
        """标准化注意力权重到[0,1]范围"""
        if not attention_weights:
            return []
        
        weights = np.array(attention_weights)
        if weights.max() == weights.min():
            return np.ones_like(weights)
        
        return (weights - weights.min()) / (weights.max() - weights.min())
    
    def create_attention_colored_mol(self, smiles, attention_weights, size=(400, 400)):
        """创建带有注意力权重的彩色分子图"""
        try:
            # 生成分子对象
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 标准化注意力权重
            norm_weights = self.normalize_attention_weights(attention_weights)
            
            # 创建颜色映射
            cmap = plt.get_cmap('YlOrRd')  # 黄-橙-红色映射，越红表示注意力越高
            
            # 为每个原子分配颜色
            atom_colors = {}
            for i, atom in enumerate(mol.GetAtoms()):
                if i < len(norm_weights):
                    # 将注意力权重映射到颜色
                    color = cmap(norm_weights[i])
                    # 转换为RGB元组
                    rgb_color = tuple(int(c * 255) for c in color[:3])
                    atom_colors[atom.GetIdx()] = rgb_color
            
            # 创建分子绘图对象
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            
            # 设置绘图参数
            drawer.drawOptions().additionalAtomLabelPadding = 0.1
            drawer.drawOptions().bondLineWidth = 2.0
            drawer.drawOptions().padding = 0.1
            
            # 绘制分子
            drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), 
                              highlightAtomColors=atom_colors)
            drawer.FinishDrawing()
            
            return drawer.GetDrawingText()
            
        except Exception as e:
            print(f"创建注意力权重分子图失败: {e}")
            return None
    
    def save_attention_mol_image(self, smiles, attention_weights, job_id, idx, size=(400, 400)):
        """保存带有注意力权重的分子图，返回文件路径"""
        try:
            # 生成图像数据
            img_data = self.create_attention_colored_mol(smiles, attention_weights, size)
            if img_data is None:
                return "/static/img/default_molecule.png"
            
            # 生成文件名
            filename = f"{job_id}_attn_{idx}.png"
            filepath = self.output_dir / filename
            
            # 保存图像
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            # 返回静态文件路径
            return f"/static/tmp/{filename}"
            
        except Exception as e:
            print(f"保存注意力权重分子图失败: {e}")
            return "/static/img/default_molecule.png"
    
    def create_attention_heatmap(self, smiles, attention_weights, job_id, idx, size=(400, 200)):
        """创建注意力权重热力图"""
        try:
            if not attention_weights:
                return "/static/img/default_molecule.png"
            
            # 标准化权重
            norm_weights = self.normalize_attention_weights(attention_weights)
            
            # 创建热力图 - 调整为正方形尺寸，更适合横向布局
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # 创建颜色映射
            cmap = plt.get_cmap('YlOrRd')
            
            # 绘制热力图
            im = ax.imshow([norm_weights], cmap=cmap, aspect='auto', 
                          vmin=0, vmax=1, interpolation='nearest')
            
            # 设置坐标轴
            ax.set_xticks(range(len(norm_weights)))
            ax.set_xticklabels([f'Atom {i+1}' for i in range(len(norm_weights))], 
                              rotation=45, ha='right', fontsize=10)
            ax.set_yticks([])
            ax.set_title(f'Attention Weights', fontsize=11, pad=15)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label('Weight', fontsize=9)
            
            # 在热力图上添加数值标签
            for i, weight in enumerate(norm_weights):
                ax.text(i, 0, f'{weight:.2f}', ha='center', va='center', 
                       color='white' if weight > 0.5 else 'black', fontweight='bold', fontsize=9)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            filename = f"{job_id}_heatmap_{idx}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return f"/static/tmp/{filename}"
            
        except Exception as e:
            print(f"创建注意力权重热力图失败: {e}")
            return "/static/img/default_molecule.png"
    
    def create_combined_visualization(self, smiles, attention_weights, job_id, idx):
        """创建组合可视化（分子图+热力图）"""
        try:
            # 生成分子图
            mol_image = self.save_attention_mol_image(smiles, attention_weights, job_id, idx)
            
            # 生成热力图
            heatmap_image = self.create_attention_heatmap(smiles, attention_weights, job_id, idx)
            
            return {
                'molecule_image': mol_image,
                'heatmap_image': heatmap_image,
                'attention_stats': {
                    'max_weight': max(attention_weights) if attention_weights else 0,
                    'min_weight': min(attention_weights) if attention_weights else 0,
                    'mean_weight': np.mean(attention_weights) if attention_weights else 0,
                    'std_weight': np.std(attention_weights) if attention_weights else 0
                }
            }
            
        except Exception as e:
            print(f"创建组合可视化失败: {e}")
            return {
                'molecule_image': "/static/img/default_molecule.png",
                'heatmap_image': "/static/img/default_molecule.png",
                'attention_stats': {}
            }
