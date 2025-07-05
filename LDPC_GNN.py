#!/usr/bin/env python3
"""
融合GATv2注意力机制与显式V2C/C2V消息传递的LDPC神经译码器
保留GATv2的注意力优势，同时明确区分两种消息传递方向
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from typing import Optional, Tuple, Dict, List


class LinearAttentionLayer(nn.Module):
    """线性注意力层，降低计算复杂度从O(n²)到O(n)"""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.feature_map_dim = 64
        self.feature_map = nn.Linear(self.head_dim, self.feature_map_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def elu_feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        Q = self.elu_feature_map(self.feature_map(Q))
        K = self.elu_feature_map(self.feature_map(K))

        KV = torch.einsum('bhnd,bhnf->bhdf', K, V)
        Z = K.sum(dim=2)

        QKV = torch.einsum('bhnd,bhdf->bhnf', Q, KV)
        QZ = torch.einsum('bhnd,bhd->bhn', Q, Z).unsqueeze(-1)

        out = QKV / (QZ + 1e-6)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        return self.layer_norm(x + out)


class ExplicitV2CLayer(nn.Module):
    """显式的变量节点到校验节点消息传递层，使用GATv2注意力"""

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # V2C专用的GATv2层
        self.v2c_gatv2 = GATv2Conv(
            hidden_dim, hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=8,  # 边特征维度
            concat=False,
            bias=False
        )

        # LLR融合层
        self.llr_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for LLR
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 度数感知门控
        self.degree_embed = nn.Embedding(100, 16)
        self.degree_gate = nn.Sequential(
            nn.Linear(16 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, node_degrees: torch.Tensor,
                llr_features: torch.Tensor, var_node_mask: torch.Tensor) -> torch.Tensor:
        """
        执行V2C消息传递
        x: 所有节点特征
        edge_index: V2C边索引
        edge_attr: 边特征
        node_degrees: 节点度数
        llr_features: LLR特征（仅变量节点）
        var_node_mask: 变量节点掩码
        """
        # 对变量节点融合LLR信息
        x_with_llr = x.clone()
        if var_node_mask.any():
            var_features = x[var_node_mask]
            fused_features = self.llr_fusion(
                torch.cat([var_features, llr_features], dim=-1)
            )
            x_with_llr[var_node_mask] = fused_features

        # 使用GATv2进行V2C消息传递
        x_v2c = self.v2c_gatv2(x_with_llr, edge_index, edge_attr)

        # 度数感知门控
        if node_degrees is not None:
            degrees_clamped = torch.clamp(node_degrees, 0, 99).long()
            degree_features = self.degree_embed(degrees_clamped)

            gate_input = torch.cat([x_v2c, degree_features], dim=-1)
            gate_weights = self.degree_gate(gate_input)

            x_v2c = x_v2c * gate_weights

        # 层归一化
        return self.layer_norm(x_v2c)


class ExplicitC2VLayer(nn.Module):
    """显式的校验节点到变量节点消息传递层，使用GATv2注意力"""

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # C2V专用的GATv2层
        self.c2v_gatv2 = GATv2Conv(
            hidden_dim, hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=8,
            concat=False,
            bias=False
        )

        # 校验节点特征变换（模拟BP中的tanh操作）
        self.check_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # 类似BP中的tanh
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 度数感知门控
        self.degree_embed = nn.Embedding(100, 16)
        self.degree_gate = nn.Sequential(
            nn.Linear(16 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, node_degrees: torch.Tensor,
                check_node_mask: torch.Tensor) -> torch.Tensor:
        """
        执行C2V消息传递
        x: 所有节点特征
        edge_index: C2V边索引
        edge_attr: 边特征
        node_degrees: 节点度数
        check_node_mask: 校验节点掩码
        """
        # 变换校验节点特征
        x_transformed = x.clone()
        if check_node_mask.any():
            check_features = x[check_node_mask]
            transformed_features = self.check_transform(check_features)
            x_transformed[check_node_mask] = transformed_features

        # 使用GATv2进行C2V消息传递
        x_c2v = self.c2v_gatv2(x_transformed, edge_index, edge_attr)

        # 度数感知门控
        if node_degrees is not None:
            degrees_clamped = torch.clamp(node_degrees, 0, 99).long()
            degree_features = self.degree_embed(degrees_clamped)

            gate_input = torch.cat([x_c2v, degree_features], dim=-1)
            gate_weights = self.degree_gate(gate_input)

            x_c2v = x_c2v * gate_weights

        # 层归一化
        return self.layer_norm(x_c2v)


class BidirectionalBPLayer(nn.Module):
    """双向BP层，整合V2C和C2V消息传递"""

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1,
                 use_linear_attention: bool = False):
        super().__init__()

        self.use_linear_attention = use_linear_attention

        # V2C和C2V消息传递层
        self.v2c_layer = ExplicitV2CLayer(hidden_dim, heads, dropout)
        self.c2v_layer = ExplicitC2VLayer(hidden_dim, heads, dropout)

        # 节点更新层（使用GRU）
        self.var_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.check_update = nn.GRUCell(hidden_dim, hidden_dim)

        # 可选的线性注意力
        if use_linear_attention:
            self.linear_attention = LinearAttentionLayer(hidden_dim, heads, dropout)

        # 残差权重
        self.v2c_residual = nn.Parameter(torch.tensor(0.1))
        self.c2v_residual = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, v2c_edge_index: torch.Tensor,
                c2v_edge_index: torch.Tensor, edge_attr: torch.Tensor,
                node_degrees: torch.Tensor, llr_features: torch.Tensor,
                var_node_mask: torch.Tensor, check_node_mask: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        执行一轮完整的V2C→C2V消息传递
        """
        # 保存原始特征用于残差连接
        x_original = x.clone()

        # Step 1: V2C消息传递
        x_v2c = self.v2c_layer(
            x, v2c_edge_index, edge_attr[:len(v2c_edge_index[0])],
            node_degrees, llr_features, var_node_mask
        )

        # 更新校验节点
        check_features_old = x[check_node_mask]
        check_features_new = self.check_update(
            x_v2c[check_node_mask], check_features_old
        )
        x[check_node_mask] = check_features_new

        # V2C残差连接
        x = x + self.v2c_residual * x_original

        # Step 2: C2V消息传递
        x_c2v = self.c2v_layer(
            x, c2v_edge_index, edge_attr[len(v2c_edge_index[0]):],
            node_degrees, check_node_mask
        )

        # 更新变量节点
        var_features_old = x[var_node_mask]
        var_features_new = self.var_update(
            x_c2v[var_node_mask], var_features_old
        )
        x[var_node_mask] = var_features_new

        # C2V残差连接
        x = x + self.c2v_residual * x_original

        # 可选的线性注意力增强
        if self.use_linear_attention and batch is not None:
            batch_size = batch.max().item() + 1
            max_nodes = (batch == 0).sum().item()

            x_seq = torch.zeros(batch_size, max_nodes, x.size(-1), device=x.device)
            for i in range(batch_size):
                mask = batch == i
                x_seq[i, :mask.sum()] = x[mask]

            x_seq = self.linear_attention(x_seq)

            x_linear = torch.zeros_like(x)
            for i in range(batch_size):
                mask = batch == i
                x_linear[mask] = x_seq[i, :mask.sum()]

            x = (x + x_linear) / 2

        return x


class ImprovedSoftSyndromeProcessor(nn.Module):
    """改进的软综合征处理器"""

    def __init__(self, hidden_dim: int, num_check_nodes: int):
        super().__init__()
        self.num_check_nodes = num_check_nodes
        self.hidden_dim = hidden_dim

        self.syndrome_proj = nn.Linear(num_check_nodes, hidden_dim // 2)

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.syndrome_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, node_features: torch.Tensor, bit_probs: torch.Tensor,
                H: torch.Tensor, var_node_mask: torch.Tensor) -> torch.Tensor:
        batch_size = bit_probs.size(0)
        device = bit_probs.device

        soft_syndrome = self._compute_soft_syndrome_stable(bit_probs, H)

        syndrome_features = self.syndrome_proj(soft_syndrome)

        num_nodes_per_graph = node_features.size(0) // batch_size
        syndrome_expanded = syndrome_features.unsqueeze(1).expand(
            batch_size, num_nodes_per_graph, -1
        ).reshape(-1, self.hidden_dim // 2)

        output = node_features.clone()
        if var_node_mask.any():
            var_features = node_features[var_node_mask]
            var_syndrome = syndrome_expanded[var_node_mask]

            fused_input = torch.cat([var_features, var_syndrome], dim=-1)
            enhanced_features = self.fusion_layer(fused_input)

            output[var_node_mask] = (1 - self.syndrome_weight) * var_features + \
                                    self.syndrome_weight * enhanced_features

        return output

    def _compute_soft_syndrome_stable(self, bit_probs: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        llr = torch.log((bit_probs + 1e-8) / (1 - bit_probs + 1e-8))
        llr = torch.clamp(llr, -10, 10)

        tanh_half_llr = torch.tanh(0.5 * llr)

        soft_syndrome = torch.matmul(H.unsqueeze(0).float(), tanh_half_llr.unsqueeze(-1))
        soft_syndrome = torch.tanh(0.5 * soft_syndrome).squeeze(-1)

        soft_syndrome_prob = 0.5 * (1 - soft_syndrome)

        return soft_syndrome_prob


class ResidualLLRConnection(nn.Module):
    """残差LLR连接"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.llr_proj = nn.Linear(1, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, node_features: torch.Tensor, original_llr: torch.Tensor,
                var_node_mask: torch.Tensor) -> torch.Tensor:
        if not var_node_mask.any():
            return node_features

        output = node_features.clone()
        var_features = node_features[var_node_mask]
        llr_features = self.llr_proj(original_llr.unsqueeze(-1))

        gate_input = torch.cat([var_features, llr_features], dim=-1)
        gate_weights = self.gate(gate_input)

        output[var_node_mask] = gate_weights * llr_features + (1 - gate_weights) * var_features

        return output


class DynamicIterationScheduler(nn.Module):
    """动态迭代调度器"""

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        self.num_layers = num_layers

        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def get_layer_importance(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        x_pooled = x.mean(dim=0, keepdim=True)
        layer_info = torch.tensor([layer_idx / self.num_layers], device=x.device).unsqueeze(0)
        importance_input = torch.cat([x_pooled, layer_info], dim=-1)
        importance = self.importance_predictor(importance_input)
        return importance.squeeze()


class GATv2ExplicitBPDecoder(nn.Module):
    """融合GATv2与显式V2C/C2V消息传递的LDPC译码器"""

    def __init__(self, n: int, m: int, hidden_dim: int = 64,
                 num_layers: int = 5, heads: int = 4,
                 dropout: float = 0.1, use_linear_attention: bool = True,
                 use_dynamic_scheduling: bool = True):
        super().__init__()

        self.n = n
        self.m = m
        self.k = n - m
        self.num_nodes = n + m
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.use_dynamic_scheduling = use_dynamic_scheduling

        self.var_node_feat_dim = 3
        self.check_node_feat_dim = 2

        # 初始嵌入层
        self.var_embed = nn.Sequential(
            nn.Linear(self.var_node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.check_embed = nn.Sequential(
            nn.Linear(self.check_node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 边特征嵌入
        self.edge_feat_dim = 2
        self.edge_embed = nn.Linear(self.edge_feat_dim, 8)

        # 双向BP层（融合GATv2）
        self.bp_layers = nn.ModuleList()
        for i in range(num_layers):
            use_linear = use_linear_attention and (i % 2 == 1)
            self.bp_layers.append(
                BidirectionalBPLayer(
                    hidden_dim, heads, dropout,
                    use_linear_attention=use_linear
                )
            )

        # 软综合征处理
        self.syndrome_processor = ImprovedSoftSyndromeProcessor(hidden_dim, m)

        # 残差LLR连接
        self.residual_llr = ResidualLLRConnection(hidden_dim)

        # 动态迭代调度
        if use_dynamic_scheduling:
            self.iteration_scheduler = DynamicIterationScheduler(num_layers, hidden_dim)

        # 输出层
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 辅助输出头
        self.syndrome_predictor = nn.Linear(hidden_dim, m)

    def build_tanner_graph(self, H: torch.Tensor, batch_size: int = 1):
        """构建Tanner图，分别生成V2C和C2V边"""
        device = H.device
        m, n = H.shape

        rows, cols = torch.where(H == 1)

        # V2C边：变量节点到校验节点
        v2c_edge_index = torch.stack([cols, rows + n])

        # C2V边：校验节点到变量节点
        c2v_edge_index = torch.stack([rows + n, cols])

        # 所有边（用于特征）
        edge_index = torch.cat([v2c_edge_index, c2v_edge_index], dim=1)

        num_edges_v2c = v2c_edge_index.size(1)
        edge_type = torch.cat([
            torch.zeros(num_edges_v2c, device=device),  # V2C
            torch.ones(num_edges_v2c, device=device)  # C2V
        ]).long()

        var_degrees = H.sum(dim=0)
        check_degrees = H.sum(dim=1)
        all_degrees = torch.cat([var_degrees, check_degrees])

        edge_features = []
        edge_features.append(edge_type.float().unsqueeze(1))

        src_degrees = all_degrees[edge_index[0]]
        dst_degrees = all_degrees[edge_index[1]]
        connection_strength = 2.0 / (src_degrees + dst_degrees)
        edge_features.append(connection_strength.unsqueeze(1))

        edge_attr = torch.cat(edge_features, dim=1)

        if batch_size > 1:
            batch_v2c_edge_index = []
            batch_c2v_edge_index = []
            batch_edge_attr = []
            batch_degrees = []

            for b in range(batch_size):
                offset = b * self.num_nodes
                batch_v2c_edge_index.append(v2c_edge_index + offset)
                batch_c2v_edge_index.append(c2v_edge_index + offset)
                batch_edge_attr.append(edge_attr)
                batch_degrees.append(all_degrees)

            v2c_edge_index = torch.cat(batch_v2c_edge_index, dim=1)
            c2v_edge_index = torch.cat(batch_c2v_edge_index, dim=1)
            edge_attr = torch.cat(batch_edge_attr)
            node_degrees = torch.cat(batch_degrees)
        else:
            node_degrees = all_degrees

        return v2c_edge_index, c2v_edge_index, edge_attr, node_degrees

    def init_node_features(self, llr: torch.Tensor, H: torch.Tensor):
        """初始化节点特征"""
        batch_size = llr.size(0)
        device = llr.device

        var_degrees = H.sum(dim=0).float()
        check_degrees = H.sum(dim=1).float()

        var_features = []
        var_features.append(llr)
        var_features.append(torch.tanh(torch.abs(llr) / 2))
        var_features.append(torch.sign(llr))
        var_features = torch.stack(var_features, dim=-1)

        check_features = []
        check_degrees_norm = (check_degrees / check_degrees.max()).unsqueeze(0).expand(batch_size, -1)
        check_features.append(check_degrees_norm)
        check_features.append(torch.ones(batch_size, self.m, device=device))
        check_features = torch.stack(check_features, dim=-1)

        return var_features, check_features

    def forward(self, llr: torch.Tensor, H: torch.Tensor,
                snr_estimate: Optional[torch.Tensor] = None):
        """前向传播"""
        batch_size = llr.size(0)
        device = llr.device

        if H.device != device:
            H = H.to(device)

        # 构建分离的V2C和C2V边
        v2c_edge_index, c2v_edge_index, edge_attr, node_degrees = self.build_tanner_graph(H, batch_size)

        # 初始化特征
        var_features, check_features = self.init_node_features(llr, H)

        # 嵌入
        var_embedded = self.var_embed(var_features.view(-1, self.var_node_feat_dim))
        check_embedded = self.check_embed(check_features.view(-1, self.check_node_feat_dim))

        # 合并节点特征
        x = torch.cat([
            var_embedded.view(batch_size, self.n, self.hidden_dim),
            check_embedded.view(batch_size, self.m, self.hidden_dim)
        ], dim=1).view(-1, self.hidden_dim)

        # 嵌入边特征
        edge_attr = self.edge_embed(edge_attr)

        # 批次信息
        batch = torch.arange(batch_size, device=device).repeat_interleave(self.num_nodes)

        # 节点掩码
        var_node_mask = torch.zeros(batch_size * self.num_nodes, dtype=torch.bool, device=device)
        check_node_mask = torch.zeros(batch_size * self.num_nodes, dtype=torch.bool, device=device)
        for b in range(batch_size):
            start_idx = b * self.num_nodes
            var_node_mask[start_idx:start_idx + self.n] = True
            check_node_mask[start_idx + self.n:start_idx + self.num_nodes] = True

        # 原始LLR（用于残差连接）
        original_llr_flat = llr.view(-1)
        llr_features = original_llr_flat.unsqueeze(-1)

        # 双向BP迭代
        for i in range(self.num_layers):
            # 动态调度
            if self.use_dynamic_scheduling:
                importance = self.iteration_scheduler.get_layer_importance(x, i)
                if importance < 0.3 and i > 0:
                    continue

            # 执行V2C→C2V消息传递
            x = self.bp_layers[i](
                x, v2c_edge_index, c2v_edge_index, edge_attr,
                node_degrees, llr_features, var_node_mask, check_node_mask, batch
            )

            # 软综合征处理（每3层一次）
            if i % 3 == 2 and i < self.num_layers - 1:
                with torch.no_grad():
                    var_feats = x[var_node_mask]
                    temp_logits = self.output_mlp(var_feats).squeeze(-1)
                    temp_probs = torch.sigmoid(temp_logits.view(batch_size, self.n))

                x = self.syndrome_processor(x, temp_probs, H, var_node_mask)

            # 残差LLR连接（每2层一次）
            if i % 2 == 1:
                x = self.residual_llr(x, original_llr_flat, var_node_mask)

        # 最终输出
        var_feats = x[var_node_mask]
        logits = self.output_mlp(var_feats).squeeze(-1)
        bit_probs = torch.sigmoid(logits.view(batch_size, self.n))

        # 辅助输出
        check_feats = x[check_node_mask].view(batch_size, self.m, -1)
        syndrome_logits = self.syndrome_predictor(check_feats.mean(dim=1))

        noise_estimate = snr_estimate if snr_estimate is not None else torch.zeros(batch_size, 1, device=device)

        return bit_probs, syndrome_logits, noise_estimate

    def decode_adaptive(self, llr: torch.Tensor, H: torch.Tensor,
                        min_iters: int = 3, max_iters: int = 10,
                        early_stop_threshold: float = 0.95) -> torch.Tensor:
        """自适应迭代解码"""
        with torch.no_grad():
            batch_size = llr.size(0)
            device = llr.device

            if H.device != device:
                H = H.to(device)

            best_decoded = None
            best_syndrome_weight = float('inf')
            previous_weight = float('inf')

            current_llr = llr.clone()
            momentum = 0.9

            for iteration in range(max_iters):
                bit_probs, _, _ = self.forward(current_llr, H)
                decoded = (bit_probs > 0.5).float()

                syndrome = torch.matmul(H.unsqueeze(0), decoded.unsqueeze(-1)) % 2
                syndrome_weight = syndrome.squeeze(-1).sum(dim=1)

                total_weight = syndrome_weight.sum().item()
                if total_weight < best_syndrome_weight:
                    best_decoded = decoded.clone()
                    best_syndrome_weight = total_weight

                if torch.all(syndrome_weight == 0) and iteration >= min_iters:
                    break

                if iteration > 0 and iteration > min_iters:
                    if best_syndrome_weight == 0 or \
                            (previous_weight - total_weight) / (previous_weight + 1e-10) < 0.01:
                        break

                previous_weight = total_weight

                if iteration < max_iters - 1:
                    soft_llr = torch.log((bit_probs + 1e-8) / (1 - bit_probs + 1e-8))
                    soft_llr = torch.clamp(soft_llr, -10, 10)

                    llr_update = momentum * current_llr + (1 - momentum) * soft_llr

                    step_size = 0.5 * (1 - iteration / max_iters)
                    current_llr = (1 - step_size) * current_llr + step_size * llr_update

            return best_decoded if best_decoded is not None else decoded

    def compute_loss(self, bit_probs: torch.Tensor, true_bits: torch.Tensor,
                     H: torch.Tensor, syndrome_logits: Optional[torch.Tensor] = None,
                     noise_estimate: Optional[torch.Tensor] = None,
                     true_noise_level: Optional[torch.Tensor] = None):
        """计算损失函数"""
        device = bit_probs.device

        bit_probs = bit_probs.float()
        true_bits = true_bits.float()
        H = H.float()

        bce_loss = F.binary_cross_entropy(
            torch.clamp(bit_probs, 1e-7, 1 - 1e-7),
            true_bits,
            reduction='mean'
        )

        syndrome_loss = torch.tensor(0.0, device=device)
        if syndrome_logits is not None:
            true_syndrome = torch.matmul(H.unsqueeze(0), true_bits.unsqueeze(-1)) % 2
            true_syndrome = true_syndrome.squeeze(-1).float()
            syndrome_loss = F.binary_cross_entropy_with_logits(
                syndrome_logits, true_syndrome, reduction='mean'
            )

        pred_syndrome = torch.matmul(H.unsqueeze(0), bit_probs.unsqueeze(-1))
        soft_syndrome_reg = torch.mean((pred_syndrome % 1.0) ** 2)

        entropy_reg = -torch.mean(
            bit_probs * torch.log(bit_probs + 1e-8) +
            (1 - bit_probs) * torch.log(1 - bit_probs + 1e-8)
        )

        total_loss = (
                1.0 * bce_loss +
                0.2 * syndrome_loss +
                0.1 * soft_syndrome_reg +
                0.01 * entropy_reg
        )

        loss_components = {
            'bce': bce_loss.item(),
            'syndrome': syndrome_loss.item(),
            'soft_syndrome': soft_syndrome_reg.item(),
            'entropy': entropy_reg.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_components


def create_gatv2_explicit_bp_model(H: np.ndarray, **kwargs) -> GATv2ExplicitBPDecoder:
    """创建融合GATv2与显式BP的LDPC译码器"""
    m, n = H.shape
    model = GATv2ExplicitBPDecoder(n=n, m=m, **kwargs)
    return model


def test_gatv2_explicit_bp():
    """测试融合模型"""
    print("=== 测试融合GATv2与显式V2C/C2V的LDPC译码器 ===")

    H = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1]
    ])

    model = create_gatv2_explicit_bp_model(
        H,
        hidden_dim=32,
        num_layers=4,
        heads=2,
        dropout=0.1,
        use_linear_attention=True,
        use_dynamic_scheduling=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    batch_size = 4
    n = H.shape[1]
    test_llr = torch.randn(batch_size, n).to(device)
    H_tensor = torch.tensor(H, dtype=torch.float32).to(device)

    try:
        with torch.no_grad():
            bit_probs, syndrome_logits, _ = model(test_llr, H_tensor)
            print("✓ 前向传播成功")
            print(f"  输出形状: {bit_probs.shape}")

            decoded = model.decode_adaptive(test_llr, H_tensor, min_iters=2, max_iters=5)
            print("✓ 自适应解码成功")

            true_bits = torch.randint(0, 2, (batch_size, n)).float().to(device)
            loss, loss_components = model.compute_loss(
                bit_probs, true_bits, H_tensor,
                syndrome_logits=syndrome_logits
            )
            print("✓ 损失计算成功")
            print(f"  损失组件: {loss_components}")

            # 检查综合征
            syndrome = torch.matmul(H_tensor.unsqueeze(0), decoded.unsqueeze(-1)) % 2
            syndrome_weight = syndrome.squeeze(-1).sum(dim=1)
            print(f"  综合征权重: {syndrome_weight}")

        print("\n融合模型测试通过！")
        print("\n主要特点:")
        print("1. 显式分离V2C和C2V消息传递")
        print("2. 每个方向使用专门的GATv2注意力机制")
        print("3. 保留度数感知和线性注意力特性")
        print("4. V2C融合LLR信息，C2V使用tanh变换")
        print("5. 双向消息传递后使用GRU更新节点状态")

        return True

    except Exception as e:
        print(f"\n✗ 模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_gatv2_explicit_bp()