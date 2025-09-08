import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self,
                 task='classification', num_layer=5, emb_dim=300, feat_dim=512,
                 drop_ratio=0, pool='mean', pred_n_layer=2, pred_act='relu',
                 fingerprint_list=None,
                 ecfp_bits=2048,
                 maccs_bits=167,
                 ap_bits=2048,
                 ext_bits=2048,
                 torsion_bits=2048, avalon_bits=1024,

                 fp_hidden_dim=512
                 ):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        # 换成全局注意力池化了
        # if pool == 'mean':
        #     self.pool = global_mean_pool
        # elif pool == 'max':
        #     self.pool = global_max_pool
        # elif pool == 'add':
        #     self.pool = global_add_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)


        # --- 全局注意力池化 ---=============================================================
        # gate_nn: 输入 emb_dim -> 输出 1 -> sigmoid 得到注意力分数
        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(emb_dim, 1),
                nn.Sigmoid()
            )
        )
        # # 将注意力池化后的向量映射到 feat_dim
        # self.feat_lin = nn.Linear(emb_dim, feat_dim)




        # —— 新增：节点打分（attention）MLP ——
        # 简单起见用单层 Linear -> 输出标量
        self.node_score_lin = nn.Linear(emb_dim, 1)  # 输入 h_i (emb_dim)，输出 s_i 标量

        # —— 动态计算总指纹长度 —— #
        total_bits = 0
        if 'ecfp' in fingerprint_list: total_bits += ecfp_bits
        if 'maccs' in fingerprint_list: total_bits += maccs_bits
        if 'ap' in fingerprint_list: total_bits += ap_bits
        if 'ext' in fingerprint_list: total_bits += ext_bits

        if 'torsion' in fingerprint_list: total_bits += torsion_bits
        if 'avalon' in fingerprint_list: total_bits += avalon_bits

        self.fp_total_bits = total_bits
        self.fp_hidden_dim = fp_hidden_dim



        # 把高维拼接后的指纹先映射到 fp_hidden_dim

        # ========（2）在这里对 pred_head 里的所有 Linear 调用 reset_parameters() ========
        def init_linear(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        # self.pred_head.apply(init_linear)
        # 把高维拼接后的指纹先映射到 fp_hidden_dim

        self.fp_mlp = nn.Sequential(
            nn.Linear(self.fp_total_bits, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, fp_hidden_dim),
            nn.ReLU(inplace=True)
        )
        # self.fp_lin = nn.Linear(self.fp_total_bits, self.fp_hidden_dim)
        self.fp_mlp.apply(init_linear)


        # ---------------- 建立预测头（Pred Head） ----------------
        # 拼接后总维度 = feat_dim + fp_hidden_dim
        # fused_dim = feat_dim
        # fused_dim =  fp_hidden_dim
        fused_dim = feat_dim + fp_hidden_dim


        if self.task == 'classification':
            out_dim = 2
        elif self.task == 'regression':
            out_dim = 1

        self.pred_n_layer = max(1, pred_n_layer)



        if pred_act == 'relu':
            pred_head = [
                nn.Linear(fused_dim, fused_dim // 2),
                nn.ReLU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(fused_dim // 2, fused_dim // 2),
                    nn.ReLU(inplace=True),
                ])
# 保存向量（神经指纹） 输入 机器学习 算法 和其他机器学习的分子指纹rdkfp，）  比较 aupr，acc结果
            pred_head.append(nn.Linear(fused_dim // 2, out_dim))
        else:
            raise ValueError('Undefined activation function')

        # pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
        self.pred_head = nn.Sequential(*pred_head)

        # ========（2）在这里对 pred_head 里的所有 Linear 调用 reset_parameters() ========
        def init_linear(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.pred_head.apply(init_linear)


############################  门控

        # gate_h: projects fused features to gating weights for h
        self.gate_h = nn.Linear(feat_dim + fp_hidden_dim, feat_dim)
        # gate_fp: projects fused features to gating weights for fp_mapped
        self.gate_fp = nn.Linear(feat_dim + fp_hidden_dim, fp_hidden_dim)
        self.gate_h.apply(init_linear)
        self.gate_fp.apply(init_linear)
        # —— 修改1：bias 初始为 -1，让 sigmoid(bias)≈0.27，梯度更活跃 ——
        nn.init.constant_(self.gate_h.bias, -1.0)
        nn.init.constant_(self.gate_fp.bias, -1.0)




    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training) # [batch_size, emb_dim]
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training) # [batch_size, emb_dim]

        # h = self.pool(h, data.batch)  # [batch_size, emb_dim]
        # --- 全局注意力池化 -----------------------==================
        node_attn = self.att_pool.gate_nn(h).squeeze(1)
        h = self.att_pool(h, data.batch)  # [batch_size, emb_dim]

        h = self.feat_lin(h) # [batch_size, feat_dim]

        # --------------- 指纹映射与拼接 ------------------
        # data.fp 形状应为 [batch_size, fp_total_bits]（从 Dataset 而来）
        fp = data.fp.view(-1, self.fp_total_bits)  # 确保是 [batch_size, fp_total_bits]
        fp_mapped = F.relu(self.fp_mlp(fp))  # 先映射：[batch_size, fp_hidden_dim]



        # 2) 拼接得到 z  11111111111111111111111111111111111
        z = torch.cat([h, fp_mapped], dim=1)  # [batch_size, feat_dim_without_fp + fp_hidden_dim]

        # 3) 通过 gate_h, gate_fp 得到两个 “0~1” 之间的门值
        gate_h = torch.sigmoid(self.gate_h(z))  # [batch_size, feat_dim_without_fp]
        gate_fp = torch.sigmoid(self.gate_fp(z))  # [batch_size, fp_hidden_dim]

        # 4) 对原始特征加权
        h_weighted = gate_h * h  # [batch_size, feat_dim_without_fp]
        fp_mapped_weighted = gate_fp * fp_mapped  # [batch_size, fp_hidden_dim]

        # 门控后拼接
        gated_z = torch.cat([h_weighted, fp_mapped_weighted], dim=1)
        gate_vec = torch.cat([gate_h, gate_fp], dim=1)
        # 残差融合：fused = raw_z + gate_vec * (gated_z - raw_z)
        fused = z + gate_vec * (gated_z - z)

        # fused = torch.cat([h_weighted, fp_mapped_weighted], dim=1)  # 形状 [batch_size, feat_dim + fp_hidden_dim]

# fused 计算pca或者 tsne可视化

# 生成6种描述符  保存画测试集合的图


        # 将 GIN 特征（h）与指纹特征（fp_mapped）拼接
        # fused = torch.cat([h, fp_mapped], dim=1)  # 形状 [batch_size, feat_dim + fp_hidden_dim]
        # -------------------------------------------------
        out = self.pred_head(fused)
        # out = self.pred_head(z)
        # out =self.pred_head(h)
        # out = self.pred_head(fp_mapped)


        # return h, self.pred_head(h)
        return h, out,node_attn

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
