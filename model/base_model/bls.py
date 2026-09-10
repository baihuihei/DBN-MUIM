import math

import torch
import torch.nn as nn


class HyTripleFusion(nn.Module):
    """
    三路 BLS 宽度融合 — 堆叠式层间传递。
    接收 seq, user, ad 三路输入，分别映射后做乘性交互和自身增强。
    """
    def __init__(self, seq_dim, user_dim, ad_dim, num_feat_nodes=128, num_cross_nodes=128, out_dim=64, num_layers=1):
        super().__init__()
        self.num_layers = max(1, int(num_layers))

        # 三路特征节点映射
        self.seq_feature_mapper = nn.Linear(seq_dim, num_feat_nodes)
        self.user_feature_mapper = nn.Linear(user_dim, num_feat_nodes)
        self.ad_feature_mapper = nn.Linear(ad_dim, num_feat_nodes)

        # 乘性交互分支 (3组)
        self.seq_ad_cross_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes) for _ in range(self.num_layers)
        ])
        self.seq_user_cross_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes) for _ in range(self.num_layers)
        ])
        self.user_ad_cross_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes) for _ in range(self.num_layers)
        ])

        # 自身增强分支 (3组)
        self.seq_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes) for _ in range(self.num_layers)
        ])
        self.user_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes) for _ in range(self.num_layers)
        ])
        self.ad_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes) for _ in range(self.num_layers)
        ])

        # 层间状态更新 (输入 = 6 * num_cross_nodes)
        self.update_layers = nn.ModuleList([
            nn.Linear(6 * num_cross_nodes, num_feat_nodes) for _ in range(self.num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(num_feat_nodes) for _ in range(self.num_layers)
        ])
        self.layer_gates = nn.Parameter(torch.full((self.num_layers,), 0.1))

        # 输出层
        total_width = (3 * num_feat_nodes) + (6 * num_cross_nodes)
        self.readout_layer = nn.Linear(total_width, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, seq_input, user_input, ad_input, return_intermediates=False):
        Z_seq = torch.relu(self.seq_feature_mapper(seq_input))
        Z_user = torch.relu(self.user_feature_mapper(user_input))
        Z_ad = torch.relu(self.ad_feature_mapper(ad_input))

        H_seq_ad = None
        H_seq_user = None
        H_user_ad = None
        H_seq = None
        H_user = None
        H_ad = None

        for layer_idx in range(self.num_layers):
            interacted_seq_ad = Z_seq * Z_ad
            H_seq_ad = torch.tanh(self.seq_ad_cross_layers[layer_idx](interacted_seq_ad))

            interacted_seq_user = Z_seq * Z_user
            H_seq_user = torch.tanh(self.seq_user_cross_layers[layer_idx](interacted_seq_user))

            interacted_user_ad = Z_user * Z_ad
            H_user_ad = torch.tanh(self.user_ad_cross_layers[layer_idx](interacted_user_ad))

            H_seq = torch.relu(self.seq_layers[layer_idx](Z_seq))
            H_user = torch.relu(self.user_layers[layer_idx](Z_user))
            H_ad = torch.relu(self.ad_layers[layer_idx](Z_ad))

            update = torch.relu(self.update_layers[layer_idx](
                torch.cat([H_seq_ad, H_seq_user, H_user_ad, H_seq, H_user, H_ad], dim=-1)
            ))
            gate = torch.sigmoid(self.layer_gates[layer_idx])

            Z_seq = self.norm_layers[layer_idx](Z_seq + gate * update)
            Z_user = self.norm_layers[layer_idx](Z_user + gate * update)
            Z_ad = self.norm_layers[layer_idx](Z_ad + gate * update)

        Broad_state = torch.cat([
            Z_seq, Z_user, Z_ad,
            H_seq_ad, H_seq_user, H_user_ad,
            H_seq, H_user, H_ad
        ], dim=-1)

        Broad_state = torch.nan_to_num(Broad_state, nan=0.0, posinf=1e4, neginf=-1e4)
        output = self.readout_layer(Broad_state)

        if return_intermediates:
            return output, {
                "Z_seq": Z_seq, "Z_user": Z_user, "Z_ad": Z_ad,
                "cross_seq_ad": H_seq_ad, "cross_seq_user": H_seq_user, "cross_user_ad": H_user_ad,
                "self_seq": H_seq, "self_user": H_user, "self_ad": H_ad,
            }
        return output


class HyBroadFusion(nn.Module):
    def __init__(self, seq_dim, non_seq_dim, num_feat_nodes=128, num_cross_nodes=128, out_dim=64, num_layers=1):
        super().__init__()
        self.num_layers = max(1, int(num_layers))
        
        # 1. 对应全局特征生成：宽泛特征节点映射 (取代压缩 token)
        self.seq_feature_mapper = nn.Linear(seq_dim, num_feat_nodes)
        if non_seq_dim is None or non_seq_dim <= 0:
            self.non_seq_feature_mapper = nn.LazyLinear(num_feat_nodes)
        else:
            self.non_seq_feature_mapper = nn.Linear(non_seq_dim, num_feat_nodes)

        # 2. 可配置层数的宽度融合块
        self.cross_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes)
            for _ in range(self.num_layers)
        ])
        self.seq_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes)
            for _ in range(self.num_layers)
        ])
        self.non_seq_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes)
            for _ in range(self.num_layers)
        ])
        self.update_layers = nn.ModuleList([
            nn.Linear(3 * num_cross_nodes, num_feat_nodes)
            for _ in range(self.num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(num_feat_nodes)
            for _ in range(self.num_layers)
        ])
        self.layer_gates = nn.Parameter(torch.full((self.num_layers,), 0.1))
        
        # 3. 对应最终增强与层级输出：全局扁平拼接
        total_width = (num_feat_nodes * 2) + (num_cross_nodes * 3)
        self.readout_layer = nn.Linear(total_width, out_dim)

    def reset_parameters(self):
        for name, module in self.named_children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, seq_input, non_seq_input, return_intermediates=False):
        # Step 1: 映射到宽泛节点映射空间
        Z_seq = torch.relu(self.seq_feature_mapper(seq_input))
        Z_non_seq = torch.relu(self.non_seq_feature_mapper(non_seq_input))
        g = torch.zeros_like(Z_seq)

        # Step 2: 多层宽度融合
        H_cross = None
        H_seq = None
        H_non_seq = None
        for layer_idx in range(self.num_layers):
            seq_ctx = Z_seq + g
            non_seq_ctx = Z_non_seq + g

            # 用非序列特征作为'门/Query'去增强序列特征
            interacted_Z = seq_ctx * non_seq_ctx  # 乘性交互
            H_cross = torch.tanh(self.cross_layers[layer_idx](interacted_Z))

            # 自身增强分支
            H_seq = torch.relu(self.seq_layers[layer_idx](seq_ctx))
            H_non_seq = torch.relu(self.non_seq_layers[layer_idx](non_seq_ctx))

            # 层间状态更新（残差 + 归一化）
            update = torch.relu(self.update_layers[layer_idx](torch.cat([H_cross, H_seq, H_non_seq], dim=-1)))
            gate = torch.sigmoid(self.layer_gates[layer_idx])
            g = self.norm_layers[layer_idx](g + gate * update)

        Z_seq_final = Z_seq + g
        Z_non_seq_final = Z_non_seq + g
        
        # Step 3: 扁平拼出超级宽度向量 (Mixer 融合的平替)
        Broad_state = torch.cat([
            Z_seq_final,  # 纯序列记忆
            Z_non_seq_final,  # 纯非序列记忆
            H_cross,    # 序列-非序列交叉模式
            H_seq,      # 序列高阶模式
            H_non_seq   # 非序列高阶模式
        ], dim=-1)

        Broad_state = torch.nan_to_num(Broad_state, nan=0.0, posinf=1e4, neginf=-1e4)
        output = self.readout_layer(Broad_state)

        if return_intermediates:
            return output, {
                "Z_seq": Z_seq_final, "Z_non_seq": Z_non_seq_final,
                "cross_seq_non_seq": H_cross,
                "self_seq": H_seq, "self_non_seq": H_non_seq,
            }
        return output


# ── 可扩展交叉组注册表 (宽度阶梯) ─────────────────────────────
# name -> (A_source, A_dim_mult, B_source, B_dim_mult)
# A/B 维度以 D 为单位；forward 中通过 feat_pool 取源张量。
# 增加宽度只需在 config 的 bls_extra_cross_pairs 里加一个名字。
EXTRA_CROSS_REGISTRY = {
    "user_id_ad_id":        ("user_id",          1.0, "ad_id",    1.0),
    "rt_att_ad_cate":       ("rt_att",           2.0, "ad_cate",  1.0),
    "user_attr_ad_loc":     ("user_attr",        2.0, "ad_loc",   2.0),
    "user_loc_ad_cate":     ("user_loc",         1.5, "ad_cate",  1.0),
    "uni_seq_ad_id":        ("uni_seq_att_v2",   2.0, "ad_id",    1.0),
    "rt_att_user_loc":      ("rt_att",           2.0, "user_loc", 1.5),
    "rt_att_out_ad_cate":   ("rt_att_out",       2.0, "ad_cate",  1.0),
    "uni_seq_out_user_attr": ("uni_seq_att_out_v2", 2.0, "user_attr", 2.0),
}


class ComplexManualCross(nn.Module):
    """
    复杂人工交叉融合模块 — 参考 SimplifiedBLSFusion 设计思想。

    设计思想：
    ─────────────────────────────────────────────────────────────
    1) 手动特征分解：
       seq (8*D) → [rt_att_out(2*D), uni_seq_att_out_v2(2*D), uni_seq_att_v2(2*D), rt_att(2*D)]
       non_seq (8.5*D) → [ad(4*D), user(4.5*D)]
       ad → [ad_id(D), ad cate(D), ad_city(D), ad_prov(D)]
       user → [user_id(D), age(D), gender(D), prov(D/2), city(D/2), level(D)]

    2) 四组 BLS 风格配对交叉 (每层独立门控残差状态):
       - Short:  rt_att × rt_att_out        (实时注意力实时交叉)
       - Long:   uni_seq_att_v2 × uni_seq_att_out_v2 (通用序列交叉)
       - AttrCate: user_attr × item cate    (用户属性-物品类别)
       - Loc:    user_loc × ad_loc          (用户位置-广告位置)

    3) 层次化融合：
       Group 1: Short + Long → Merge → Z_seq
       Group 2: AttrCate + Loc → Add → static_out
       Group 3: Z_seq × static_out → Final → final_out

    4) BLS 交叉增强 (每组每层):
       输入 A, B → 特征节点(已映射) → 拼接增强(frozen W) → Hadamard交叉(frozen W)
       → 拼接 → update_layer → 门控残差 → 更新状态 g

    5) 宽度侧: ad_id + user_id + final_out → wide_logits (2维)
    """

    def __init__(self, seq_dim: int, non_seq_dim: int, D: int = 32,
                 node_dim: int = 32, out_dim: int = 64, num_layers: int = 1,
                 activate: str = "tanh", extra_pairs=None):
        super().__init__()
        self.D = D
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.num_layers = max(1, int(num_layers))
        self.activate = activate

        # ═══════════════════════════════════════════
        # 输入投影层 (各子特征 → node_dim)
        # ═══════════════════════════════════════════
        # Group 1 Short: rt_att(2*D) vs rt_att_out(2*D)
        self.short_proj_A = nn.Linear(2 * D, node_dim)
        self.short_proj_B = nn.Linear(2 * D, node_dim)
        # Group 1 Long: uni_seq_att_v2(2*D) vs uni_seq_att_out_v2(2*D)
        self.long_proj_A = nn.Linear(2 * D, node_dim)
        self.long_proj_B = nn.Linear(2 * D, node_dim)
        # Merge: concat(short_out, long_out) → node_dim
        self.seq_merge = nn.Linear(2 * node_dim, node_dim)

        # Group 2 AttrCate: user_attr(age+gender=2*D) vs item cate(D)
        self.attr_proj = nn.Linear(2 * D, node_dim)
        self.cate_proj = nn.Linear(D, node_dim)
        # Group 2 Loc: user_loc(prov+city+level=1.5*D) vs ad_loc(city+prov=2*D)
        # user loc = prov(D/2) + city(D/2) + level(D/2) = 1.5*D
        self.user_loc_proj = nn.Linear(3 * (D // 2), node_dim)
        self.ad_loc_proj = nn.Linear(2 * D, node_dim)

        # ═══════════════════════════════════════════
        # 各组 BLS 随机增强节点权重 () — node_dim 空间
        # ═══════════════════════════════════════════
        # 每组: W_enhance (2*node_dim → node_dim), W_cross (node_dim → node_dim)
        for group in ['short', 'long', 'attr_cate', 'loc', 'final']:
            W_enh = torch.randn(2 * node_dim, node_dim) / math.sqrt(2 * node_dim)
            b_enh = torch.zeros(node_dim)
            W_cross = torch.randn(node_dim, node_dim) / math.sqrt(node_dim)
            b_cross = torch.zeros(node_dim)
            self.register_buffer(f'{group}_W_enhance', W_enh)
            self.register_buffer(f'{group}_b_enhance', b_enh)
            self.register_buffer(f'{group}_W_cross', W_cross)
            self.register_buffer(f'{group}_b_cross', b_cross)

        # ═══════════════════════════════════════════
        # Group 1: Short branch — 门控残差 (L层, 独立 g_s)
        # ═══════════════════════════════════════════
        self.short_update = nn.ModuleList([
            nn.Linear(4 * node_dim, node_dim) for _ in range(self.num_layers)
        ])
        self.short_norm = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(self.num_layers)
        ])
        self.short_gates = nn.Parameter(torch.full((self.num_layers,), 0.1))

        # ═══════════════════════════════════════════
        # Group 1: Long branch — 门控残差 (L层, 独立 g_l)
        # ═══════════════════════════════════════════
        self.long_update = nn.ModuleList([
            nn.Linear(4 * node_dim, node_dim) for _ in range(self.num_layers)
        ])
        self.long_norm = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(self.num_layers)
        ])
        self.long_gates = nn.Parameter(torch.full((self.num_layers,), 0.1))

        # ═══════════════════════════════════════════
        # Group 2: AttrCate branch — 门控残差 (L层, 独立 g_a)
        # ═══════════════════════════════════════════
        self.attr_cate_update = nn.ModuleList([
            nn.Linear(4 * node_dim, node_dim) for _ in range(self.num_layers)
        ])
        self.attr_cate_norm = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(self.num_layers)
        ])
        self.attr_cate_gates = nn.Parameter(torch.full((self.num_layers,), 0.1))

        # ═══════════════════════════════════════════
        # Group 2: Loc branch — 门控残差 (L层, 独立 g_b)
        # ═══════════════════════════════════════════
        self.loc_update = nn.ModuleList([
            nn.Linear(4 * node_dim, node_dim) for _ in range(self.num_layers)
        ])
        self.loc_norm = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(self.num_layers)
        ])
        self.loc_gates = nn.Parameter(torch.full((self.num_layers,), 0.1))

        # ═══════════════════════════════════════════
        # Group 3: Final branch — 门控残差 (L层, 独立 g_f, node_dim 空间)
        # ═══════════════════════════════════════════
        self.final_update = nn.ModuleList([
            nn.Linear(4 * node_dim, node_dim) for _ in range(self.num_layers)
        ])
        self.final_norm = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(self.num_layers)
        ])
        self.final_gates = nn.Parameter(torch.full((self.num_layers,), 0.1))

        # Final 投影到 out_dim (BLS 交叉输出 4*node_dim → 聚合到 out_dim)
        self.final_proj = nn.Linear(4 * node_dim, out_dim)

        # ═══════════════════════════════════════════
        # 可扩展交叉组 (宽度阶梯) — 每组与 Loc 组同构:
        #   proj_A / proj_B → 门控残差 BLS 交叉 → 输出 node_dim
        # 组名由 config bls_extra_cross_pairs 指定，每组独立 gate。
        # ═══════════════════════════════════════════
        self.extra_pairs = list(extra_pairs) if extra_pairs else []
        for name in self.extra_pairs:
            if name not in EXTRA_CROSS_REGISTRY:
                raise ValueError(
                    f"Unknown extra cross pair '{name}'. "
                    f"Available: {list(EXTRA_CROSS_REGISTRY)}"
                )
            A_src, A_mult, B_src, B_mult = EXTRA_CROSS_REGISTRY[name]
            A_dim, B_dim = int(A_mult * self.D), int(B_mult * self.D)
            setattr(self, f"{name}_proj_A", nn.Linear(A_dim, node_dim))
            setattr(self, f"{name}_proj_B", nn.Linear(B_dim, node_dim))
            self.register_buffer(
                f"{name}_W_enhance",
                torch.randn(2 * node_dim, node_dim) / math.sqrt(2 * node_dim),
            )
            self.register_buffer(f"{name}_b_enhance", torch.zeros(node_dim))
            self.register_buffer(
                f"{name}_W_cross", torch.randn(node_dim, node_dim) / math.sqrt(node_dim)
            )
            self.register_buffer(f"{name}_b_cross", torch.zeros(node_dim))
            setattr(self, f"{name}_update", nn.ModuleList([
                nn.Linear(4 * node_dim, node_dim) for _ in range(self.num_layers)
            ]))
            setattr(self, f"{name}_norm", nn.ModuleList([
                nn.LayerNorm(node_dim) for _ in range(self.num_layers)
            ]))
        self.extra_gates = nn.ParameterDict({
            name: nn.Parameter(torch.full((self.num_layers,), 0.1))
            for name in self.extra_pairs
        })

        # ═══════════════════════════════════════════
        # 输出层: 拼接 Z_seq + static_out + final_out (+ 扩展组) → readout
        # ═══════════════════════════════════════════
        total_width = (2 * node_dim) + out_dim + node_dim * len(self.extra_pairs)
        self.readout = nn.Linear(total_width, out_dim)

        # 宽度侧预测 (wide)
        wide_dim = 2 * D + out_dim
        self.wide_layer = nn.Linear(wide_dim, 2)

        self.reset_parameters()

    # ────────────────────────────────────────────────
    # BLS 交叉增强: 拼接增强 + Hadamard 交叉增强
    # ────────────────────────────────────────────────
    def _bls_cross(self, feat_A, feat_B, W_enh, b_enh, W_cross, b_cross):
        """BLS 风格交叉增强，返回拼接后的宽度向量 [B, 4*dim]"""
        # 拼接增强
        concat_feat = torch.cat([feat_A, feat_B], dim=-1)
        if self.activate == "tanh":
            H_enhance = torch.tanh(concat_feat @ W_enh + b_enh)
            # Hadamard 交叉增强
            H_cross = torch.tanh((feat_A * feat_B) @ W_cross + b_cross)
        else:
            H_enhance = torch.relu(concat_feat @ W_enh + b_enh)
            H_cross = torch.relu((feat_A * feat_B) @ W_cross + b_cross)
        # [feat_A, feat_B, H_enhance, H_cross] — 4 个组件
        return torch.cat([feat_A, feat_B, H_enhance, H_cross], dim=-1)

    def reset_parameters(self):
        for name, module in self.named_children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        nn.init.zeros_(self.wide_layer.weight)
        nn.init.zeros_(self.wide_layer.bias)

    def get_gates(self, sigmoid: bool = True):
        """
        返回所有交叉组的门控值，用于分析"每路/每层"的激活程度。
        - 基础 5 组: short / long / attr_cate / loc / final
        - 扩展组:     extra_* (来自 bls_extra_cross_pairs)
        sigmoid=True 返回 σ(gate) ∈ (0,1)，接近 0 说明该组/层被模型忽略。
        """
        gates = {}
        for name in ["short", "long", "attr_cate", "loc", "final"]:
            g = getattr(self, f"{name}_gates")
            gates[name] = (torch.sigmoid(g) if sigmoid else g).detach().cpu().tolist()
        for name in self.extra_pairs:
            g = self.extra_gates[name]
            gates[f"extra_{name}"] = (
                torch.sigmoid(g) if sigmoid else g
            ).detach().cpu().tolist()
        return gates

    def forward(self, seq_input: torch.Tensor, non_seq_input: torch.Tensor, return_intermediates=False):
        """
        Args:
            seq_input:     [B, 8*D]    序列特征
            non_seq_input: [B, 8.5*D]  非序列特征 (ad + user)
        Returns:
            output:        [B, out_dim] 融合后输出
        """
        N = self.node_dim
        D = self.D

        # ═══════════════════════════════════════════
        # Step 1: 手动特征分解
        # ═══════════════════════════════════════════
        # seq: [rt_att_out(2D), uni_seq_att_out_v2(2D), uni_seq_att_v2(2D), rt_att(2D)]
        split_size = 2 * D
        rt_att_out, uni_seq_att_out_v2, uni_seq_att_v2, rt_att = \
            seq_input.split(split_size, dim=1)

        # non_seq: [ad(4D), user(4.5D)]
        ad_dim = 4 * D
        ad_part = non_seq_input[:, :ad_dim]
        user_part = non_seq_input[:, ad_dim:]

        # ad: [ad_id(D), ad cate(D), ad_city(D), ad_prov(D)]
        ad_id_emb = ad_part[:, :D]
        ad_cate_emb = ad_part[:, D:2*D]
        ad_city_emb = ad_part[:, 2*D:3*D]
        ad_prov_emb = ad_part[:, 3*D:4*D]

        # user: [user_id(D), age(D), gender(D), prov(D/2), city(D/2), level(D)]
        user_id_emb = user_part[:, :D]
        user_feats = user_part[:, D:]
        age_emb = user_feats[:, :D]
        gender_emb = user_feats[:, D:2*D]
        half_D = D // 2
        prov_emb = user_feats[:, 2*D:2*D + half_D]
        city_emb = user_feats[:, 2*D + half_D:2*D + D]
        level_emb = user_feats[:, 2*D + D:]

        # ═══════════════════════════════════════════
        # Step 2: 一次性投影到 node_dim
        # ═══════════════════════════════════════════
        # Group 1
        short_A = torch.relu(self.short_proj_A(rt_att))
        short_B = torch.relu(self.short_proj_B(rt_att_out))
        long_A = torch.relu(self.long_proj_A(uni_seq_att_v2))
        long_B = torch.relu(self.long_proj_B(uni_seq_att_out_v2))

        # Group 2
        user_attr = torch.relu(self.attr_proj(torch.cat([age_emb, gender_emb], dim=-1)))
        item_cate = torch.relu(self.cate_proj(ad_cate_emb))
        user_loc = torch.relu(
            self.user_loc_proj(torch.cat([prov_emb, city_emb, level_emb], dim=-1))
        )
        ad_loc = torch.relu(
            self.ad_loc_proj(torch.cat([ad_city_emb, ad_prov_emb], dim=-1))
        )

        # ═══════════════════════════════════════════
        # Group 1: Short (g_s) — BLS 交叉, L层门控残差堆叠
        # ═══════════════════════════════════════════
        g_s = torch.zeros_like(short_A)
        for layer_idx in range(self.num_layers):
            bls_out = self._bls_cross(
                short_A + g_s, short_B + g_s,
                getattr(self, f'short_W_enhance'),
                getattr(self, f'short_b_enhance'),
                getattr(self, f'short_W_cross'),
                getattr(self, f'short_b_cross'),
            )
            gate = torch.sigmoid(self.short_gates[layer_idx])
            update = torch.relu(self.short_update[layer_idx](bls_out))
            g_s = self.short_norm[layer_idx](g_s + gate * update)
        short_out = short_A + short_B + g_s  # [B, node_dim]

        # ═══════════════════════════════════════════
        # Group 1: Long (g_l) — BLS 交叉, L层门控残差堆叠
        # ═══════════════════════════════════════════
        g_l = torch.zeros_like(long_A)
        for layer_idx in range(self.num_layers):
            bls_out = self._bls_cross(
                long_A + g_l, long_B + g_l,
                getattr(self, f'long_W_enhance'),
                getattr(self, f'long_b_enhance'),
                getattr(self, f'long_W_cross'),
                getattr(self, f'long_b_cross'),
            )
            gate = torch.sigmoid(self.long_gates[layer_idx])
            update = torch.relu(self.long_update[layer_idx](bls_out))
            g_l = self.long_norm[layer_idx](g_l + gate * update)
        long_out = long_A + long_B + g_l  # [B, node_dim]

        # ── Merge: concat(short_out, long_out) → Z_seq ──
        Z_seq = self.seq_merge(torch.cat([short_out, long_out], dim=-1))  # [B, node_dim]

        # ═══════════════════════════════════════════
        # Group 2: AttrCate (g_a) — BLS 交叉, L层门控残差堆叠
        # ═══════════════════════════════════════════
        g_a = torch.zeros_like(user_attr)
        for layer_idx in range(self.num_layers):
            bls_out = self._bls_cross(
                user_attr + g_a, item_cate + g_a,
                getattr(self, 'attr_cate_W_enhance'),
                getattr(self, 'attr_cate_b_enhance'),
                getattr(self, 'attr_cate_W_cross'),
                getattr(self, 'attr_cate_b_cross'),
            )
            gate = torch.sigmoid(self.attr_cate_gates[layer_idx])
            update = torch.relu(self.attr_cate_update[layer_idx](bls_out))
            g_a = self.attr_cate_norm[layer_idx](g_a + gate * update)
        attr_cate_out = user_attr + item_cate + g_a  # [B, node_dim]

        # ═══════════════════════════════════════════
        # Group 2: Loc (g_b) — BLS 交叉, L层门控残差堆叠
        # ═══════════════════════════════════════════
        g_b = torch.zeros_like(user_loc)
        for layer_idx in range(self.num_layers):
            bls_out = self._bls_cross(
                user_loc + g_b, ad_loc + g_b,
                getattr(self, f'loc_W_enhance'),
                getattr(self, f'loc_b_enhance'),
                getattr(self, f'loc_W_cross'),
                getattr(self, f'loc_b_cross'),
            )
            gate = torch.sigmoid(self.loc_gates[layer_idx])
            update = torch.relu(self.loc_update[layer_idx](bls_out))
            g_b = self.loc_norm[layer_idx](g_b + gate * update)
        loc_out = user_loc + ad_loc + g_b  # [B, node_dim]

        # ── Add: attr_cate_out + loc_out → static_out ──
        static_out = attr_cate_out + loc_out  # [B, node_dim]

        # ═══════════════════════════════════════════
        # Group 3: Final (g_f) — Z_seq × static_out, BLS 交叉, L层门控残差堆叠
        # ═══════════════════════════════════════════
        # Final group 在 node_dim 空间做 BLS 交叉，最后投影到 out_dim
        g_f = torch.zeros_like(Z_seq)  # [B, node_dim]
        final_bls_out = None
        for layer_idx in range(self.num_layers):
            final_bls_out = self._bls_cross(
                Z_seq + g_f, static_out + g_f,
                self.final_W_enhance, self.final_b_enhance,
                self.final_W_cross, self.final_b_cross,
            )  # [B, 4*node_dim]
            gate = torch.sigmoid(self.final_gates[layer_idx])
            update = torch.relu(self.final_update[layer_idx](final_bls_out))  # [B, node_dim]
            g_f = self.final_norm[layer_idx](g_f + gate * update)
        final_out = self.final_proj(final_bls_out)  # [B, 4*node_dim] → [B, out_dim]

        # ═══════════════════════════════════════════
        # 扩展交叉组 (宽度阶梯): 每组输出 node_dim，扁平拼进 broad_state
        # ═══════════════════════════════════════════
        extra_outs = {}
        if self.extra_pairs:
            # 从分解后的原始子特征构造特征源池
            feat_pool = {
                "rt_att": rt_att, "rt_att_out": rt_att_out,
                "uni_seq_att_v2": uni_seq_att_v2,
                "uni_seq_att_out_v2": uni_seq_att_out_v2,
                "ad_id": ad_id_emb, "ad_cate": ad_cate_emb,
                "ad_city": ad_city_emb, "ad_prov": ad_prov_emb,
                "user_id": user_id_emb, "age": age_emb, "gender": gender_emb,
                "prov": prov_emb, "city": city_emb, "level": level_emb,
                "user_attr": torch.cat([age_emb, gender_emb], dim=-1),
                "user_loc": torch.cat([prov_emb, city_emb, level_emb], dim=-1),
                "ad_loc": torch.cat([ad_city_emb, ad_prov_emb], dim=-1),
            }
            for name in self.extra_pairs:
                A_src, _, B_src, _ = EXTRA_CROSS_REGISTRY[name]
                A = torch.relu(getattr(self, f"{name}_proj_A")(feat_pool[A_src]))
                B = torch.relu(getattr(self, f"{name}_proj_B")(feat_pool[B_src]))
                g = torch.zeros_like(A)
                for layer_idx in range(self.num_layers):
                    bls_out = self._bls_cross(
                        A + g, B + g,
                        getattr(self, f"{name}_W_enhance"),
                        getattr(self, f"{name}_b_enhance"),
                        getattr(self, f"{name}_W_cross"),
                        getattr(self, f"{name}_b_cross"),
                    )
                    gate = torch.sigmoid(self.extra_gates[name][layer_idx])
                    update = torch.relu(getattr(self, f"{name}_update")[layer_idx](bls_out))
                    g = getattr(self, f"{name}_norm")[layer_idx](g + gate * update)
                extra_outs[name] = A + B + g  # [B, node_dim]

        # ═══════════════════════════════════════════
        # Step 3: Readout — 扁平拼接输出
        # ═══════════════════════════════════════════
        broad_state = torch.cat(
            [Z_seq, static_out, final_out] + [extra_outs[n] for n in self.extra_pairs],
            dim=-1,
        )
        broad_state = torch.nan_to_num(broad_state, nan=0.0, posinf=1e4, neginf=-1e4)
        output = self.readout(broad_state)  # [B, out_dim]

        if return_intermediates:
            return output, {
                "short_out": short_out, "long_out": long_out,
                "attr_cate_out": attr_cate_out, "loc_out": loc_out,
                "static_out": static_out, "final_out": final_out,
                **{f"extra_{k}": v for k, v in extra_outs.items()},
            }
        return output
