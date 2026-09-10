import sys
import os
import logging

import torch
import torch.distributed as dist

from model.base_model.simtier import cosine_simtier_list
from model.base_model.layers import multi_head_att, multi_head_att_v2, fc_repeats
from model.base_model.bls import HyBroadFusion, HyTripleFusion, ComplexManualCross
from model.base_model.waid import DeepFMBlock, DCNv2Block, GDCNBlock, CINBlock, FCNBlock, FINALBlock, FiBiNETBlock, SFGBlock, FinalMLPBlock, AFMBlock, AFNBlock, FmFMBlock

from utils.utils import clip_prop, write_info_to_file

class MUSE_DIN(torch.nn.Module):
    def __init__(self, args=dict(), D=32, RT_STEPS=50, UNI_STEPS=50):
        super().__init__()
        self.args = args
        self.D = D
        self.RT_STEPS = RT_STEPS
        self.UNI_STEPS = UNI_STEPS
        
        # simtier
        self.simtier_eps_list = self.args.get("simtier_eps_list", None)
        self.all_seq_image_res = cosine_simtier_list(
            steps=[RT_STEPS, UNI_STEPS],
            n_dim=128,
            scope_list=["rt", "uni_v2"],
            eps_list=[0.1, 0.1],
            dim_list=[1, 1],
            simtier_eps_lists=[self.simtier_eps_list, self.simtier_eps_list],
        )

        self.realtime_att = multi_head_att(
            2 * self.D, 2 * self.D, [2 * self.D], [2 * self.D]
        )
        self.attn_score_cross = self.args.get("use_sata", True)  # whether to use SA-TA
        self.uni_att_v2 = multi_head_att_v2(
            2 * self.D, 2 * self.D, [2 * self.D], [2 * self.D], attn_score_cross=self.attn_score_cross
        )

        # self.uni_att_item = multi_head_att_v2(
        #     self.D, self.D, [self.D], [self.D], attn_score_cross=self.attn_score_cross
        # )
        # self.uni_att_cate = multi_head_att_v2(
        #     self.D, self.D, [self.D], [self.D], attn_score_cross=self.attn_score_cross
        # )
        
        self.bls_out_dim = 64

        # Ablation study controls
        self.use_target_attn = self.args.get("use_target_attn", True)
        self.simtier_mode = self.args.get("simtier_mode", "multi")  # "multi", "single", "none"

        # SimTier feature dimensions per scope
        # Dynamically computed from the resolution list.
        # Default 5-resolution: eps=[0.1,0.2,0.3,0.5,0.6] → 22+12+8+6+4=52
        # Single resolution: eps=[0.1] → 22
        if self.simtier_mode == "none":
            self.simtier_feat_dim_per_scope = 0
        else:
            eps_for_dim = self.simtier_eps_list if self.simtier_eps_list else [0.1, 0.2, 0.3, 0.5, 0.6]
            if self.simtier_mode == "single":
                eps_for_dim = [eps_for_dim[0]]  # only finest
            self.simtier_feat_dim_per_scope = sum(
                2 * int(1.0 / e) + 2 for e in eps_for_dim
            )
        self.simtier_total_dim = self.simtier_feat_dim_per_scope * 2  # 2 scopes

        # SimTier dim varies by mode: multi=104, single=44, none=0
        base_din_dim = 15 * self.D + 3 * (self.D // 2) + self.simtier_total_dim
        
        self.bls_non_seq_dim = int(self.args.get("bls_non_seq_dim", 102))
        self.bls_num_layers = int(self.args.get("bls_num_layers", 3))

        self.width_method = self.args.get("width_method", "bls")
        self.bls_l2_weight = float(self.args.get("bls_l2_weight", 0.0000))

        # Configurable width-side module
        seq_dim = 8 * self.D  # 8 * 32 = 256
        # non_seq_dim = 4*D (ad: 205,206,213,214) + 3*D + 3*(D//2) (user: 129_1,130_1,130_2,130_3,130_4,130_5)
        # = 4*D + 3*D + 1.5*D = 8.5*D
        non_seq_dim = int(8.5 * self.D)
        ad_dim = 4 * self.D
        user_dim = non_seq_dim - ad_dim

        if self.width_method == "none":
            self.wide_fusion = None
            self.bls_gate = 0.0
        elif self.width_method == "bls":
            self.wide_fusion = HyBroadFusion(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                num_feat_nodes=128, 
                num_cross_nodes=128, 
                out_dim=self.bls_out_dim,
                num_layers=self.bls_num_layers
            )
            self.bls_gate = torch.nn.Parameter(torch.tensor(0.1))
        elif self.width_method == "triple_bls":
            self.wide_fusion = HyTripleFusion(
                seq_dim=seq_dim,
                user_dim=user_dim,
                ad_dim=ad_dim,
                num_feat_nodes=128,
                num_cross_nodes=128,
                out_dim=self.bls_out_dim,
                num_layers=self.bls_num_layers
            )
            self.bls_gate = torch.nn.Parameter(torch.tensor(0.1))
        elif self.width_method == "complex_manual_cross":
            self.wide_fusion = ComplexManualCross(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                D=self.D,
                node_dim=32,
                out_dim=self.bls_out_dim,
                num_layers=self.bls_num_layers,
                activate="tanh",
                extra_pairs=self.args.get("bls_extra_cross_pairs", []),
            )
            self.bls_gate = torch.nn.Parameter(torch.tensor(0.1))
        elif self.width_method == "deepfm":
            self.wide_fusion = DeepFMBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                num_fields=self.args.get("waid_num_fields", 32),
                fm_embed_dim=self.args.get("waid_fm_embed_dim", 8),
            )
            self.bls_gate = 1.0
        elif self.width_method == "dcnv2":
            self.wide_fusion = DCNv2Block(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                num_layers=self.args.get("waid_num_layers", 3),
                deep_hidden_dim=self.args.get("waid_deep_hidden_dim", 128),
            )
            self.bls_gate = 1.0
        elif self.width_method == "gdcn":
            self.wide_fusion = GDCNBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                num_layers=self.args.get("waid_num_layers", 3),
                deep_hidden_dim=self.args.get("waid_deep_hidden_dim", 128),
            )
            self.bls_gate = 1.0
        elif self.width_method == "cin":
            self.wide_fusion = CINBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                num_fields=self.args.get("waid_num_fields", 16),
                num_layers=self.args.get("waid_num_layers", 3),
                cin_hidden_dim=self.args.get("waid_cin_hidden_dim", 128),
            )
            self.bls_gate = 1.0
        elif self.width_method == "fcn":
            self.wide_fusion = FCNBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                hidden_dims=self.args.get("waid_hidden_dims", (256, 128)),
            )
            self.bls_gate = 1.0
        elif self.width_method == "final":
            self.wide_fusion = FINALBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                hidden_dim=self.args.get("waid_hidden_dim", 256),
                num_layers=self.args.get("waid_num_layers", 3),
                rank=self.args.get("waid_rank", 8),
            )
            self.bls_gate = 1.0
        elif self.width_method == "fibinet":
            self.wide_fusion = FiBiNETBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                num_fields=self.args.get("waid_num_fields", 32),
                reduction_ratio=self.args.get("waid_reduction_ratio", 4),
                bilinear_dim=self.args.get("waid_bilinear_dim", 8),
            )
            self.bls_gate = 1.0
        elif self.width_method == "sfg":
            self.wide_fusion = SFGBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                num_layers=self.args.get("waid_num_layers", 3),
            )
            self.bls_gate = 1.0
        elif self.width_method == "finalmlp":
            self.wide_fusion = FinalMLPBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                hidden_dims=self.args.get("waid_hidden_dims", (256, 128)),
                num_layers=self.args.get("waid_num_layers", 2),
            )
            self.bls_gate = 1.0
        elif self.width_method == "afm":
            self.wide_fusion = AFMBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                num_fields=self.args.get("waid_num_fields", 32),
                attention_size=self.args.get("waid_attention_size", 16),
            )
            self.bls_gate = 1.0
        elif self.width_method == "afn":
            self.wide_fusion = AFNBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                num_fields=self.args.get("waid_num_fields", 32),
                hidden_dims=self.args.get("waid_hidden_dims", (256, 128)),
            )
            self.bls_gate = 1.0
        elif self.width_method == "fmfm":
            self.wide_fusion = FmFMBlock(
                seq_dim=seq_dim,
                non_seq_dim=non_seq_dim,
                out_dim=self.bls_out_dim,
                num_fields=self.args.get("waid_num_fields", 32),
                fm_embed_dim=self.args.get("waid_fm_embed_dim", 8),
                use_pair_weights=self.args.get("waid_fmfm_pair_weights", True),
            )
            self.bls_gate = 1.0
        else:
            raise ValueError(f"Unknown width_method: {self.width_method}")

        din_dim = base_din_dim
        self.fc_tower = fc_repeats(din_dim, shape=[256, 128, 64, 2], acts=['dice', 'dice', 'dice', 'dice', 'dice', None])

        self.wide_layer = torch.nn.Linear(self.bls_out_dim, 2)
        torch.nn.init.zeros_(self.wide_layer.weight)
        torch.nn.init.zeros_(self.wide_layer.bias)
        
        self.use_aux_loss = self.args["use_aux_loss"]
        if self.use_aux_loss:
            self.fc_tower_aux = fc_repeats(self.D * 4, shape=[200, 80, 2], acts=['dice', 'dice', None])

        self.use_kl_loss = self.args.get("use_kl_loss", True)
        self.kl_loss_weight = float(self.args.get("kl_loss_weight", 0.05))
        self.kl_temperature = max(float(self.args.get("kl_temperature", 1.0)), 1e-6)
        self.kl_eps = float(self.args.get("kl_eps", 1e-8))

        self.reset_parameters()

    def reset_parameters(self): 
        for name, module in self.named_children(): 
            module.reset_parameters()
        torch.nn.init.zeros_(self.wide_layer.weight)
        torch.nn.init.zeros_(self.wide_layer.bias)

    def get_bls_gates(self, sigmoid: bool = True):
        """导出 BLS 各交叉组的门控值（ComplexManualCross 专用），供饱和分析。"""
        if self.width_method == "complex_manual_cross":
            return self.wide_fusion.get_gates(sigmoid=sigmoid)
        return None

    def forward(
        self,
        user_embs,
        ad_embs,
        uni_seq_embs,
        short_seq_fn,
        label,
        return_attn=False,
        seq_item_ids=None,
    ):
        
        item_content = torch.reshape(ad_embs[-1], [-1, 1, 128])
        ad_embs = ad_embs[:-1]

        eval_flag = (uni_seq_embs[0].requires_grad == False)

        all_seq_image_res = self.all_seq_image_res(
            item_content,
            [
                short_seq_fn[-1],
                uni_seq_embs[-1],
            ],
            [None, None],
        )

        # ── SimTier 模式控制 ──
        if self.simtier_mode == "none":
            # 不加 SimTier: 仅移除多分辨率直方图特征，保留原始 cosine 用于注意力引导
            B = all_seq_image_res[1][0].size(0)
            for scope_idx in range(len(all_seq_image_res[1])):
                all_seq_image_res[1][scope_idx] = torch.zeros(
                    B, 0, device=all_seq_image_res[1][scope_idx].device,
                    dtype=all_seq_image_res[1][scope_idx].dtype,
                )
            # 注意: all_seq_image_res[0] (cosine) 不归零，否则注意力失去多模态引导
        elif self.simtier_mode == "single":
            # 仅保留最细粒度分辨率 (eps_list[0] only)
            keep_dims = self.simtier_feat_dim_per_scope
            for scope_idx in range(len(all_seq_image_res[1])):
                feat = all_seq_image_res[1][scope_idx]
                all_seq_image_res[1][scope_idx] = feat[:, :keep_dims].contiguous()
        # "multi" 模式: 保持所有特征不变 (默认行为)

        rt_att = torch.concat(
            [
                torch.reshape(short_seq_fn[k], [-1, self.RT_STEPS, self.D])
                for k in range(2)
            ],
            dim=2,
        )
        uni_seq_att_v2 = torch.concat(
            [
                torch.reshape(uni_seq_embs[k], [-1, self.UNI_STEPS, self.D]) 
                for k in range(2)
            ], 
            dim=2
        )

        att_ad_2 = torch.reshape(
            torch.concat(ad_embs[:2], dim=1),
            [-1, 1, 2 * self.D],
        )

        # 始终计算注意力（原版 MUSE 风格）
        rt_att_out = self.realtime_att(att_ad_2, rt_att)
        uni_seq_att_out_v2 = self.uni_att_v2(att_ad_2, uni_seq_att_v2, mm_cosine=[all_seq_image_res[0][1]])

        # 消融: 不加长序列 ID 注意力 → 零化注意力输出（保留 mean pooled 序列）
        if not self.use_target_attn:
            uni_seq_att_out_v2 = torch.zeros_like(uni_seq_att_out_v2).detach()

        loss_kl = torch.zeros((), dtype=label.dtype, device=label.device)
        id_attn_dist_for_export = None
        image_attn_dist_for_export = None
        image_cosine_raw = None

        _compute_attn = (
            (self.use_kl_loss and self.use_target_attn and self.args["method"] not in ["din"])
            or return_attn
        )
        if _compute_attn:
            id_attn_score_raw = self.uni_att_v2.calc_attn_score(att_ad_2, uni_seq_att_v2)
            id_attn_dist_for_export = id_attn_score_raw / (
                id_attn_score_raw.sum(dim=1, keepdim=True) + self.kl_eps
            )

            image_cosine_raw = all_seq_image_res[0][1]
            image_attn_dist_for_export = torch.nn.functional.softmax(
                image_cosine_raw / self.kl_temperature, dim=1
            ).detach()

            if self.use_kl_loss and self.use_target_attn and self.args["method"] not in ["din"]:
                loss_kl = torch.nn.functional.kl_div(
                    torch.log(id_attn_dist_for_export + self.kl_eps),
                    image_attn_dist_for_export,
                    reduction="batchmean",
                )

        ad, user = torch.concat(ad_embs, dim=1), torch.concat(user_embs, dim=1)

        uni_seq_att_v2 = uni_seq_att_v2.mean(dim=1)
        rt_att = rt_att.mean(dim=1)

        rt_att_out = rt_att_out.squeeze(1)
        uni_seq_att_out_v2 = uni_seq_att_out_v2.squeeze(1)

        if self.args["method"] in ["din"]:
            uni_seq_att_v2 = torch.zeros_like(uni_seq_att_v2).detach()
            uni_seq_att_out_v2 = torch.zeros_like(uni_seq_att_out_v2).detach()
            all_seq_image_res[1][1] = torch.zeros_like(all_seq_image_res[1][1]).detach()

        seq_features_for_bls = torch.concat(
            [rt_att_out, uni_seq_att_out_v2, uni_seq_att_v2, rt_att], 
            dim=1
        )
        non_seq_features_for_bls = torch.concat(
            [ad, user], 
            dim=1
        )

        # Width-side fusion
        bls_intermediates = None
        if self.width_method == "none":
            wide_branch_input = torch.zeros(seq_features_for_bls.size(0), self.bls_out_dim, device=label.device)
        elif self.width_method == "triple_bls":
            # HyTripleFusion takes 3 inputs: seq, user, ad
            ad_dim = 4 * self.D
            ad_part = non_seq_features_for_bls[:, :ad_dim]
            user_part = non_seq_features_for_bls[:, ad_dim:]
            if return_attn:
                wide_cross_feature, bls_intermediates = self.wide_fusion(
                    seq_features_for_bls, user_part, ad_part, return_intermediates=True
                )
            else:
                wide_cross_feature = self.wide_fusion(seq_features_for_bls, user_part, ad_part)
            wide_cross_feature = self.bls_gate * wide_cross_feature
            wide_cross_feature = torch.nan_to_num(wide_cross_feature, nan=0.0, posinf=1e4, neginf=-1e4)
            wide_branch_input = wide_cross_feature
        elif self.width_method == "complex_manual_cross":
            # ComplexManualCross internally splits seq/non_seq and does multi-group BLS cross
            ad_dim = 4 * self.D
            ad_part = non_seq_features_for_bls[:, :ad_dim]
            user_part = non_seq_features_for_bls[:, ad_dim:]
            if return_attn:
                wide_cross_feature, bls_intermediates = self.wide_fusion(
                    seq_features_for_bls, non_seq_features_for_bls, return_intermediates=True
                )
            else:
                wide_cross_feature = self.wide_fusion(seq_features_for_bls, non_seq_features_for_bls)
            wide_cross_feature = self.bls_gate * wide_cross_feature
            wide_cross_feature = torch.nan_to_num(wide_cross_feature, nan=0.0, posinf=1e4, neginf=-1e4)
            wide_branch_input = wide_cross_feature
        else:
            # For BLS method, check seq_feature_mapper dim
            if self.width_method == "bls":
                if seq_features_for_bls.shape[1] != self.wide_fusion.seq_feature_mapper.in_features:
                    raise RuntimeError(
                        f"BLS seq dim mismatch: got {seq_features_for_bls.shape[1]}, "
                        f"expect {self.wide_fusion.seq_feature_mapper.in_features}"
                    )
            if return_attn:
                wide_cross_feature, bls_intermediates = self.wide_fusion(
                    seq_features_for_bls, non_seq_features_for_bls, return_intermediates=True
                )
            else:
                wide_cross_feature = self.wide_fusion(seq_features_for_bls, non_seq_features_for_bls)
            wide_cross_feature = self.bls_gate * wide_cross_feature
            wide_cross_feature = torch.nan_to_num(wide_cross_feature, nan=0.0, posinf=1e4, neginf=-1e4)
            wide_branch_input = wide_cross_feature

        din = torch.concat(
            [rt_att_out, uni_seq_att_out_v2, uni_seq_att_v2, rt_att, ad, user] + all_seq_image_res[1],
            dim=1
        )
        expected_fc_in = self.fc_tower.linears[0].in_features
        if din.shape[1] != expected_fc_in:
            raise RuntimeError(
                f"fc_tower input dim mismatch: got {din.shape[1]}, expect {expected_fc_in}"
            )

        deep_logits = self.fc_tower(din)
        if wide_branch_input.shape[1] != self.wide_layer.in_features:
            raise RuntimeError(
                f"wide_layer input dim mismatch: got {wide_branch_input.shape[1]}, expect {self.wide_layer.in_features}"
            )
        wide_logits = self.wide_layer(wide_branch_input)
        item_fc6 = deep_logits + wide_logits
        prop = clip_prop(
            torch.nn.functional.softmax(item_fc6, dim=-1) + 0.0000001
        )
        loss_ce = -(torch.log(prop) * label).sum(dim=1, keepdim=True)

        if self.use_aux_loss:
            din_aux = torch.cat([uni_seq_att_out_v2, uni_seq_att_v2], dim=1)
            fc_out_aux = self.fc_tower_aux(din_aux)
            prop_aux = clip_prop(
                torch.nn.functional.softmax(fc_out_aux, dim=-1) + 0.0000001
            )
            loss_aux = -(torch.log(prop_aux) * label).sum(dim=1, keepdim=True)
            total_loss = loss_ce.mean(dim=0) + loss_aux.mean(dim=0) + self.kl_loss_weight * loss_kl
        else:
            total_loss = loss_ce.mean(dim=0) + self.kl_loss_weight * loss_kl

        if self.bls_l2_weight > 0 and self.width_method == "bls":
            bls_l2_reg = sum(
                p.pow(2).sum() for name, p in self.wide_fusion.named_parameters()
                if 'weight' in name
            )
            total_loss = total_loss + self.bls_l2_weight * bls_l2_reg

        if return_attn:
            return total_loss, prop, {
                "id_attn": id_attn_dist_for_export,
                "img_attn": image_attn_dist_for_export,
                "img_cosine_raw": image_cosine_raw,
                "seq_item_ids": seq_item_ids,
                "bls_intermediates": bls_intermediates,
            }

        return total_loss, prop
    
    def save_ckpt(self, ckpt_path: str, rank=None):
        if rank is not None and rank != 0:
            return

        state_dict = {
            'all_seq_image_res': self.all_seq_image_res.state_dict(),
            'realtime_att': self.realtime_att.state_dict(),
            'uni_att_v2': self.uni_att_v2.state_dict(),
            'fc_tower': self.fc_tower.state_dict(),
            'wide_layer': self.wide_layer.state_dict(),
            'wide_fusion': self.wide_fusion.state_dict(),
            'bls_gate': self.bls_gate.detach().cpu()
        }

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(state_dict, ckpt_path)
        logging.info(f"Checkpoint saved to {ckpt_path}")
    
    def load_ckpt(self, ckpt_path: str, map_location=None):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        device_id = torch.cuda.current_device()

        if map_location is None:
            map_location = lambda storage, loc: storage.cuda(device_id)

        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.all_seq_image_res.load_state_dict(ckpt['all_seq_image_res'])
        self.realtime_att.load_state_dict(ckpt['realtime_att'])
        self.uni_att_v2.load_state_dict(ckpt['uni_att_v2'])
        self.fc_tower.load_state_dict(ckpt['fc_tower'])
        if 'wide_layer' in ckpt:
            wide_layer_state = ckpt['wide_layer']
            if (
                wide_layer_state.get('weight', None) is not None
                and wide_layer_state['weight'].shape == self.wide_layer.weight.shape
                and wide_layer_state.get('bias', None) is not None
                and wide_layer_state['bias'].shape == self.wide_layer.bias.shape
            ):
                self.wide_layer.load_state_dict(wide_layer_state)
            else:
                logging.warning(
                    f"Skip loading wide_layer due to shape mismatch."
                )
        if 'wide_fusion' in ckpt:
            self.wide_fusion.load_state_dict(ckpt['wide_fusion'], strict=False)
        if 'bls_gate' in ckpt:
            self.bls_gate.data.copy_(ckpt['bls_gate'].to(self.bls_gate.device))

        logging.info(f"[Rank {device_id}] Checkpoint loaded from {ckpt_path}")
        return self
