import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFMBlock(nn.Module):
    """
    DeepFM width-side fusion module.
    Models 1st-order (linear) and 2nd-order (pairwise FM) feature interactions.
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64, num_fields=32, fm_embed_dim=8):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.num_fields = num_fields
        self.fm_embed_dim = fm_embed_dim

        # Ensure even split into fields
        self.field_size = (total_dim + num_fields - 1) // num_fields
        self.pad_size = self.field_size * num_fields - total_dim

        # First-order term
        self.linear = nn.Linear(total_dim, out_dim)

        # Second-order embeddings per field (immediately initialized for DDP compatibility)
        self.fm_emb = nn.Parameter(torch.empty(num_fields, self.field_size, fm_embed_dim))
        nn.init.xavier_uniform_(self.fm_emb)

        # Second-order projection
        self.fm_out = nn.Linear(fm_embed_dim, out_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.fm_emb)
        nn.init.xavier_uniform_(self.fm_out.weight)
        nn.init.zeros_(self.fm_out.bias)

    def forward(self, seq_input, non_seq_input):
        x = torch.cat([seq_input, non_seq_input], dim=-1)  # (B, total_dim)

        # --- First order (use unpadded x to match linear.in_features) ---
        first_order = self.linear(x)  # (B, out_dim)

        # --- Second order (FM) ---
        # Pad if needed for even field split into num_fields
        x_pad = x
        if self.pad_size > 0:
            x_pad = F.pad(x, (0, self.pad_size))
        # x_pad: (B, num_fields * field_size) -> (B, num_fields, field_size)
        x_fields = x_pad.view(-1, self.num_fields, self.field_size)
        # Field embeddings: (B, num_fields, fm_embed_dim)
        field_emb = torch.einsum('bfd,fde->bfe', x_fields, self.fm_emb)

        # FM formula: 0.5 * ( (sum)^2 - sum(sq) )
        summed = field_emb.sum(dim=1)      # (B, fm_embed_dim)
        squared = (field_emb ** 2).sum(dim=1)  # (B, fm_embed_dim)
        pairwise = 0.5 * (summed ** 2 - squared)  # (B, fm_embed_dim)
        second_order = self.fm_out(pairwise)  # (B, out_dim)

        return first_order + second_order


class DCNv2Block(nn.Module):
    """
    DCNv2 width-side fusion module.
    Uses Cross Network with residual connections for explicit feature crossing.
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64, num_layers=3, deep_hidden_dim=128):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.total_dim = total_dim
        self.num_layers = num_layers

        # Cross layers (DCNv2 style with residual)
        self.cross_weights = nn.ModuleList([
            nn.Linear(total_dim, total_dim, bias=True) for _ in range(num_layers)
        ])

        # Optional deep MLP branch alongside cross network
        self.deep_mlp = nn.Sequential(
            nn.Linear(total_dim, deep_hidden_dim),
            nn.ReLU(),
            nn.Linear(deep_hidden_dim, deep_hidden_dim),
            nn.ReLU(),
        )

        # Combine cross + deep
        self.readout = nn.Linear(total_dim + deep_hidden_dim, out_dim)

    def reset_parameters(self):
        for m in self.cross_weights:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.deep_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        x0 = torch.cat([seq_input, non_seq_input], dim=-1)  # (B, total_dim)

        # Cross network: x_{l+1} = x0 * (W_l * x_l + b_l) + x_l
        xl = x0
        for layer in self.cross_weights:
            xl = x0 * layer(xl) + xl

        # Deep path
        h = self.deep_mlp(x0)

        # Combine and project
        combined = torch.cat([xl, h], dim=-1)
        output = self.readout(combined)
        return output


class GDCNBlock(nn.Module):
    """
    GDCN (Gated Deep Cross Network) width-side fusion module.
    Extends DCNv2 with a learnable gate per cross layer to dynamically
    control the amount of feature crossing at each layer.
    
    DCNv2:  x_{l+1} = x0 * (W_l * x_l + b_l) + x_l
    GDCN:   gate = sigmoid(W_gate * x_l + b_gate)
            x_{l+1} = x0 * (W_l * x_l * gate + b_l) + x_l
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64, num_layers=3, deep_hidden_dim=128):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.total_dim = total_dim
        self.num_layers = num_layers

        # Cross layers with gating (GDCN style)
        self.cross_weights = nn.ModuleList([
            nn.Linear(total_dim, total_dim, bias=True) for _ in range(num_layers)
        ])
        # Gate networks: learnable sigmoid gates per layer
        self.gate_networks = nn.ModuleList([
            nn.Linear(total_dim, total_dim, bias=True) for _ in range(num_layers)
        ])

        # Optional deep MLP branch alongside cross network
        self.deep_mlp = nn.Sequential(
            nn.Linear(total_dim, deep_hidden_dim),
            nn.ReLU(),
            nn.Linear(deep_hidden_dim, deep_hidden_dim),
            nn.ReLU(),
        )

        # Combine cross + deep
        self.readout = nn.Linear(total_dim + deep_hidden_dim, out_dim)

    def reset_parameters(self):
        for m in self.cross_weights:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.gate_networks:
            nn.init.xavier_uniform_(m.weight)
            # Initialize gate bias to 0 so gates start near 0.5 (sigmoid(0)=0.5)
            nn.init.zeros_(m.bias)
        for m in self.deep_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        x0 = torch.cat([seq_input, non_seq_input], dim=-1)  # (B, total_dim)
        x0 = torch.nan_to_num(x0, nan=0.0, posinf=1e4, neginf=-1e4)

        # GDCN cross network with gating:
        # gate = sigmoid(W_gate * x_l + b_gate)
        # x_{l+1} = x0 * (W_l * x_l * gate + b_l) + x_l
        xl = x0
        for cross_layer, gate_layer in zip(self.cross_weights, self.gate_networks):
            gate = torch.sigmoid(gate_layer(xl))  # (B, total_dim), values in (0, 1)
            xl = x0 * (cross_layer(xl) * gate) + xl
            xl = torch.nan_to_num(xl, nan=0.0, posinf=1e4, neginf=-1e4)

        # Deep path
        h = self.deep_mlp(x0)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

        # Combine and project
        combined = torch.cat([xl, h], dim=-1)
        output = self.readout(combined)
        return torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)


class CINBlock(nn.Module):
    """
    CIN (Compressed Interaction Network) from xDeepFM.
    Performs vector-wise feature interactions at each layer.
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64,
                 num_fields=32, num_layers=3, cin_hidden_dim=128):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.num_fields = num_fields
        self.num_layers = num_layers
        self.cin_hidden_dim = cin_hidden_dim

        # Split into fields
        self.field_size = (total_dim + num_fields - 1) // num_fields
        self.pad_size = self.field_size * num_fields - total_dim

        # CIN layers
        # Layer 0 input: (B, F*F, D) where F=num_fields, D=field_size
        # Layer k (k>=1) input: (B, H_{k-1}*F, D) where H_{k-1}=cin_hidden_dim
        self.cin_layers = nn.ModuleList()
        in_channels = num_fields * num_fields  # H_0 * F = F * F
        for _ in range(num_layers):
            out_channels = cin_hidden_dim
            self.cin_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )
            in_channels = out_channels * num_fields  # H_k * F for next layer

        # Combine all CIN layer outputs
        total_cin_out = num_layers * cin_hidden_dim
        self.readout = nn.Linear(total_cin_out, out_dim)

    def reset_parameters(self):
        for m in self.cin_layers:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        x = torch.cat([seq_input, non_seq_input], dim=-1)  # (B, total_dim)
        if self.pad_size > 0:
            x = F.pad(x, (0, self.pad_size))

        B = x.shape[0]
        # X^0: (B, F, D) where F=num_fields, D=field_size
        x_fields = x.view(B, self.num_fields, self.field_size)
        x0 = x_fields

        cin_outputs = []
        xk = x0  # X^{k-1}: (B, H_{k-1}, D), initially H_0 = F

        for layer_idx, conv in enumerate(self.cin_layers):
            H_prev = xk.shape[1]  # H_{k-1}

            # Pairwise Hadamard products: X^{k-1} ⊙ X^0
            # xk: (B, H_{k-1}, D) -> (B, H_{k-1}, 1, D)
            # x0: (B, F, D) -> (B, 1, F, D)
            # outer: (B, H_{k-1}, F, D)
            outer = xk.unsqueeze(2) * x0.unsqueeze(1)
            # Explicitly expand to avoid broadcasting ambiguity
            outer = outer.view(B, H_prev, self.num_fields, self.field_size)

            # Flatten: (B, H_{k-1} * F, D)
            outer = outer.reshape(B, H_prev * self.num_fields, self.field_size)

            # Conv1d: (B, H_{k-1}*F, D) -> (B, H_k, D)
            zk = conv(outer)

            # Sum pooling over D dimension for output: (B, H_k)
            xk_out = zk.sum(dim=-1)
            cin_outputs.append(xk_out)

            # Keep xk as (B, H_k, D) for next layer's Hadamard product
            xk = zk

        # Concatenate all CIN layer outputs
        cin_concat = torch.cat(cin_outputs, dim=-1)  # (B, num_layers * H)
        output = self.readout(cin_concat)
        return output


class FCNBlock(nn.Module):
    """
    FCN (Fully Connected Network) width-side baseline.
    Simple MLP with no explicit feature crossing.
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64, hidden_dims=(256, 128)):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(total_dim, h))
            layers.append(nn.ReLU())
            total_dim = h
        layers.append(nn.Linear(total_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, seq_input, non_seq_input):
        x = torch.cat([seq_input, non_seq_input], dim=-1)
        return self.mlp(x)


class FINALBlock(nn.Module):
    """
    FINAL (Factorized Interaction Layer) width-side fusion module.
    Extends linear layer to explicitly learn 2nd-order feature interactions
    via low-rank factorization. Stacking L layers captures 2^L degree interactions.
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64,
                 hidden_dim=256, num_layers=3, rank=8):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.layers = nn.ModuleList([
            _FINALayer(total_dim if i == 0 else hidden_dim, hidden_dim, rank)
            for i in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        for m in self.layers:
            m.reset_parameters()
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        x = torch.cat([seq_input, non_seq_input], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.readout(x)


class _FINALayer(nn.Module):
    """
    Single FINAL (Factorized Interaction Layer).
    y = W1*x + (P*x) ⊙ (Q*x) + b
    where P, Q are low-rank factor matrices.
    """
    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.P = nn.Parameter(torch.empty(in_dim, out_dim * rank))
        self.Q = nn.Parameter(torch.empty(in_dim, out_dim * rank))
        nn.init.xavier_uniform_(self.P)
        nn.init.xavier_uniform_(self.Q)
        self.out_dim = out_dim
        self.rank = rank

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.xavier_uniform_(self.P)
        nn.init.xavier_uniform_(self.Q)

    def forward(self, x):
        # First-order term
        first = self.W1(x)  # (B, out_dim)

        # Second-order factorized interaction term
        # (P*x) has shape (B, out_dim * rank), reshape to (B, out_dim, rank)
        px = (x @ self.P).view(-1, self.out_dim, self.rank)
        qx = (x @ self.Q).view(-1, self.out_dim, self.rank)
        second = (px * qx).sum(dim=-1)  # (B, out_dim)

        return first + second


class FiBiNETBlock(nn.Module):
    """
    FiBiNET (Feature Importance and Bilinear Interaction Network) width-side module.
    Two stages:
      1. SENET: squeeze-excitation to reweight feature fields
      2. Bilinear Interaction: pairwise field interactions with learnable matrices
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64,
                 num_fields=32, reduction_ratio=4, bilinear_dim=8):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.num_fields = num_fields
        self.bilinear_dim = bilinear_dim

        # Split into fields
        self.field_size = (total_dim + num_fields - 1) // num_fields
        self.pad_size = self.field_size * num_fields - total_dim

        # SENET: squeeze-excitation
        squeezed_dim = max(num_fields // reduction_ratio, 2)
        self.senet_reduce = nn.Linear(num_fields, squeezed_dim)
        self.senet_expand = nn.Linear(squeezed_dim, num_fields)

        # Bilinear interaction matrices (one shared matrix per field pair)
        # Use a single bilinear matrix for efficiency: (F, D, D_b)
        self.W_bilinear = nn.Parameter(
            torch.empty(num_fields, self.field_size, bilinear_dim)
        )
        nn.init.xavier_uniform_(self.W_bilinear)

        # Output projection
        # Total bilinear pairs: C(num_fields, 2) ≈ num_fields^2/2
        num_pairs = num_fields * (num_fields - 1) // 2
        self.readout = nn.Linear(num_pairs * bilinear_dim, out_dim)

        # Pre-compute upper triangular pair indices for vectorized computation
        idx_i, idx_j = torch.triu_indices(num_fields, num_fields, offset=1)
        self.register_buffer('pair_idx_i', idx_i, persistent=False)
        self.register_buffer('pair_idx_j', idx_j, persistent=False)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.senet_reduce.weight)
        nn.init.zeros_(self.senet_reduce.bias)
        nn.init.xavier_uniform_(self.senet_expand.weight)
        nn.init.zeros_(self.senet_expand.bias)
        nn.init.xavier_uniform_(self.W_bilinear)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        x = torch.cat([seq_input, non_seq_input], dim=-1)  # (B, total_dim)
        if self.pad_size > 0:
            x = F.pad(x, (0, self.pad_size))

        # Split into fields: (B, F, D)
        x_fields = x.view(-1, self.num_fields, self.field_size)

        # --- SENET ---
        # Global pooling over field dimension: (B, F)
        senet_input = x_fields.mean(dim=-1)
        senet_hidden = F.relu(self.senet_reduce(senet_input))
        senet_weights = torch.sigmoid(self.senet_expand(senet_hidden))
        # Reweighted features: (B, F, D)
        x_weighted = x_fields * senet_weights.unsqueeze(-1)

        # --- Bilinear Interaction ---
        # Project each field with bilinear matrix: (B, F, D_b)
        x_bilin = torch.einsum('bfd,fde->bfe', x_weighted, self.W_bilinear)

        # Vectorized pairwise interactions using pre-computed indices
        # x_bilin[:, pair_idx_i]: (B, num_pairs, D_b)
        # x_bilin[:, pair_idx_j]: (B, num_pairs, D_b)
        pair_products = x_bilin[:, self.pair_idx_i] * x_bilin[:, self.pair_idx_j]  # (B, num_pairs, D_b)
        bilinear_out = pair_products.reshape(pair_products.size(0), -1)  # (B, num_pairs * D_b)

        return self.readout(bilinear_out)


class SFGBlock(nn.Module):
    """
    SFG (Stacked Feature Gating) width-side fusion module.
    Uses stacked gating layers with residual connections to reweight features.
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64, num_layers=3):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, total_dim),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        self.readout = nn.Linear(total_dim, out_dim)

    def reset_parameters(self):
        for gate in self.gates:
            for m in gate:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        x = torch.cat([seq_input, non_seq_input], dim=-1)
        for gate in self.gates:
            g = gate(x)    # gate weights
            x = x * g + x  # gated weighting + residual
        return self.readout(x)


class FinalMLPBlock(nn.Module):
    """
    FinalMLP width-side fusion module.
    Two-stream MLP with feature interaction via element-wise product.
    
    Stream A processes seq features, Stream B processes non-seq features.
    Interaction = StreamA_out * StreamB_out (element-wise).
    Output = concat([StreamA_out, StreamB_out, Interaction]).
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64,
                 hidden_dims=(256, 128), num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        # Stream A (seq)
        layers_a = []
        in_dim = seq_dim
        for i in range(num_layers):
            h = hidden_dims[i] if i < len(hidden_dims) else hidden_dims[-1]
            layers_a.append(nn.Linear(in_dim, h))
            layers_a.append(nn.ReLU())
            in_dim = h
        self.stream_a = nn.Sequential(*layers_a)

        # Stream B (non-seq)
        layers_b = []
        in_dim = non_seq_dim
        for i in range(num_layers):
            h = hidden_dims[i] if i < len(hidden_dims) else hidden_dims[-1]
            layers_b.append(nn.Linear(in_dim, h))
            layers_b.append(nn.ReLU())
            in_dim = h
        self.stream_b = nn.Sequential(*layers_b)

        # Output: concat(stream_a, stream_b, interaction) -> out_dim
        stream_out_dim = hidden_dims[-1] if len(hidden_dims) > 0 else hidden_dims[0]
        self.readout = nn.Linear(stream_out_dim * 3, out_dim)

    def reset_parameters(self):
        for m in self.stream_a:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.stream_b:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        a_out = self.stream_a(seq_input)       # (B, H)
        b_out = self.stream_b(non_seq_input)   # (B, H)
        interact = a_out * b_out               # (B, H), element-wise product
        combined = torch.cat([a_out, b_out, interact], dim=-1)  # (B, 3*H)
        return self.readout(combined)


class AFMBlock(nn.Module):
    """
    AFM (Attentional Factorization Machine) width-side fusion module.
    Extends FM with an attention network that learns the importance of
    each pairwise feature interaction.
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64,
                 num_fields=32, attention_size=16):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.num_fields = num_fields
        self.attention_size = attention_size

        # Split into fields
        self.field_size = (total_dim + num_fields - 1) // num_fields
        self.pad_size = self.field_size * num_fields - total_dim

        # Embedding for pairwise interactions
        self.fm_emb = nn.Parameter(torch.empty(num_fields, self.field_size, attention_size))
        nn.init.xavier_uniform_(self.fm_emb)

        # Attention network: h * ReLU(W * (v_i * v_j) + b)
        self.attention = nn.Sequential(
            nn.Linear(attention_size, attention_size),
            nn.ReLU(),
            nn.Linear(attention_size, 1, bias=False),  # scalar attention weight
        )

        # Output projection
        self.readout = nn.Linear(attention_size, out_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fm_emb)
        for m in self.attention:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        x = torch.cat([seq_input, non_seq_input], dim=-1)  # (B, total_dim)
        if self.pad_size > 0:
            x = F.pad(x, (0, self.pad_size))

        B = x.shape[0]
        # Split into fields: (B, F, D)
        x_fields = x.view(B, self.num_fields, self.field_size)

        # Project to attention space: (B, F, A)
        field_emb = torch.einsum('bfd,fda->bfa', x_fields, self.fm_emb)

        # Compute all pairwise interactions with attention
        # Get upper triangular indices
        idx_i, idx_j = torch.triu_indices(self.num_fields, self.num_fields, offset=1)
        # (B, num_pairs, A)
        pair_interact = field_emb[:, idx_i] * field_emb[:, idx_j]

        # Attention weights: (B, num_pairs, 1)
        attn_scores = self.attention(pair_interact)  # (B, num_pairs, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # softmax over pairs

        # Weighted sum of pairwise interactions: (B, A)
        attended = (attn_weights * pair_interact).sum(dim=1)

        # Output
        return self.readout(attended)


class AFNBlock(nn.Module):
    """
    AFN (Adaptive Factorization Network) width-side fusion module.
    Uses Logarithmic Neural Network to learn feature interaction orders
    adaptively from data.
    
    Key idea: log(|x| + eps) -> DNN -> exp() to learn arbitrary exponents
    for feature interactions.
    
    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64,
                 num_fields=32, hidden_dims=(256, 128)):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.num_fields = num_fields

        # Split into fields
        self.field_size = (total_dim + num_fields - 1) // num_fields
        self.pad_size = self.field_size * num_fields - total_dim

        # Logarithmic transformation: learnable bias for numerical stability
        self.log_bias = nn.Parameter(torch.zeros(1))

        # Logarithmic DNN: learns exponents for each field
        # Input: num_fields * field_size (after log transform)
        # Output: num_fields (exponents per field)
        log_dims = [num_fields * self.field_size] + list(hidden_dims) + [num_fields]
        layers = []
        for i in range(len(log_dims) - 2):
            layers.append(nn.Linear(log_dims[i], log_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(log_dims[-2], log_dims[-1]))
        self.log_dnn = nn.Sequential(*layers)

        # Output projection: weighted sum of field embeddings
        self.field_weight = nn.Parameter(torch.empty(num_fields, self.field_size))
        nn.init.xavier_uniform_(self.field_weight)

        self.readout = nn.Linear(self.field_size, out_dim)

    def reset_parameters(self):
        nn.init.zeros_(self.log_bias)
        for m in self.log_dnn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.field_weight)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        x = torch.cat([seq_input, non_seq_input], dim=-1)  # (B, total_dim)
        if self.pad_size > 0:
            x = F.pad(x, (0, self.pad_size))

        B = x.shape[0]
        # Split into fields: (B, F, D)
        x_fields = x.view(B, self.num_fields, self.field_size)

        # Logarithmic transformation: log(|x| + eps)
        eps = 1e-8
        x_log = torch.log(torch.abs(x_fields) + self.log_bias.abs() + eps)  # (B, F, D)

        # Learn exponents via DNN
        x_flat = x_log.reshape(B, -1)  # (B, F*D)
        exponents = self.log_dnn(x_flat)  # (B, F)
        exponents = torch.clamp(exponents, -5.0, 5.0)  # stability

        # Apply learned exponents: |x|^p * sign(x)
        # x_fields_sign = torch.sign(x_fields)
        # x_fields_pow = torch.abs(x_fields) ** exponents.unsqueeze(-1)
        # x_transformed = x_fields_sign * x_fields_pow

        # Alternative: use log-space approach (more stable)
        # exp(exponents * log(|x|)) = |x|^exponents
        x_pow = torch.exp(exponents.unsqueeze(-1) * x_log)  # (B, F, D)
        x_transformed = torch.sign(x_fields) * x_pow

        # Weighted sum over fields: (B, D)
        out = torch.einsum('bfd,fd->bd', x_transformed, self.field_weight)

        return self.readout(out)


class FmFMBlock(nn.Module):
    """
    FmFM (Field-matrixed Factorization Machine) width-side fusion module.

    Replaces AFN's log/exp log-space network with numerically stable
    field-matrixed kernel products.

    Core idea: for each pair of fields (i, j), model their interaction via a
    field matrix M_ij (kernel product):
        y_pair = <x_i, M_ij x_j>
    For efficiency, M_ij is factorized as M_ij = U_i^T U_j (per-field shared
    projection), so:
        y_pair = <U_i x_i, U_j x_j>
    which yields the (weighted) FM formula in the projected space:
        sum_{i<j} <p_i, p_j> = 0.5 * ( ||sum_i p_i||^2 - sum_i ||p_i||^2 )
    Optionally, per-field-pair scalars w_ij (FwFM-style) refine the kernel.

    Input: seq_input (B, seq_dim), non_seq_input (B, non_seq_dim)
    Output: (B, out_dim)
    """
    def __init__(self, seq_dim, non_seq_dim, out_dim=64,
                 num_fields=32, fm_embed_dim=8, use_pair_weights=True):
        super().__init__()
        total_dim = seq_dim + non_seq_dim
        self.num_fields = num_fields
        self.fm_embed_dim = fm_embed_dim
        self.use_pair_weights = use_pair_weights

        # Ensure even split into fields
        self.field_size = (total_dim + num_fields - 1) // num_fields
        self.pad_size = self.field_size * num_fields - total_dim

        # First-order term
        self.linear = nn.Linear(total_dim, out_dim)

        # Field matrices factorized: U_i: (num_fields, fm_embed_dim, field_size)
        # M_ij ≈ U_i^T U_j  (kernel product in shared space)
        self.field_proj = nn.Parameter(
            torch.empty(num_fields, fm_embed_dim, self.field_size)
        )
        nn.init.xavier_uniform_(self.field_proj)

        # Optional per-field-pair scalar weights (FwFM-style refinement)
        n_pairs = num_fields * (num_fields - 1) // 2
        self.pair_weight = nn.Parameter(torch.ones(n_pairs))

        # Second-order projection
        self.readout = nn.Linear(fm_embed_dim, out_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.field_proj)
        nn.init.ones_(self.pair_weight)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, seq_input, non_seq_input):
        x = torch.cat([seq_input, non_seq_input], dim=-1)  # (B, total_dim)

        # --- First order ---
        first_order = self.linear(x)  # (B, out_dim)

        # --- Second order (field-matrixed interactions) ---
        x_pad = x
        if self.pad_size > 0:
            x_pad = F.pad(x, (0, self.pad_size))
        x_fields = x_pad.view(-1, self.num_fields, self.field_size)  # (B, F, D)

        # Project each field: p_i = U_i x_i  -> (B, F, k)
        proj = torch.einsum('bfd,fkd->bfk', x_fields, self.field_proj)

        if self.use_pair_weights:
            # Weighted pairwise: sum_{i<j} w_ij <p_i, p_j>
            B, n_fields, k = proj.shape
            w_mat = torch.zeros(n_fields, n_fields, device=x.device)
            idx = 0
            for i in range(n_fields):
                for j in range(i + 1, n_fields):
                    w_mat[i, j] = self.pair_weight[idx]
                    w_mat[j, i] = self.pair_weight[idx]
                    idx += 1
            proj_t = proj.permute(0, 2, 1)  # (B, k, n_fields)
            w_mat_b = w_mat.unsqueeze(0).expand(B, -1, -1)  # (B, n_fields, n_fields)
            pairwise = torch.bmm(proj_t, w_mat_b)  # (B, k, n_fields)
            pairwise = torch.bmm(pairwise, proj)  # (B, k, k)
            pairwise = torch.diagonal(pairwise, dim1=1, dim2=2)  # (B, k)
            pairwise = 0.5 * pairwise
        else:
            # Standard factorized FM: 0.5 * ( (sum p)^2 - sum(p^2) )
            summed = proj.sum(dim=1)          # (B, k)
            squared = (proj ** 2).sum(dim=1)  # (B, k)
            pairwise = 0.5 * (summed ** 2 - squared)

        second_order = self.readout(pairwise)  # (B, out_dim)

        return first_order + second_order
