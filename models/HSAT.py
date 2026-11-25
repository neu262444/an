import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mlp import MLP
from torch_scatter import scatter
from torch.nn.attention import sdpa_kernel
import hypergraph



def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class BasicTransformer(nn.Module):
    def __init__(self, hidden_channels, n_layers=1, heads=8, dropout_rate=0.1, mixer="attn"):
        super().__init__()
       
        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=hidden_channels,
                ffn_size=2 * hidden_channels,
                dropout_rate=dropout_rate,
                attention_dropout_rate=dropout_rate,
                num_heads=heads,
                mixer=mixer
            )
            for _ in range(n_layers)

        ])
        self.final_ln = nn.LayerNorm(hidden_channels)

    def reset_parameters(self):
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()

    

    def forward(self, x, attn_bias=None, raw=None, v_src="phi", skip_src="phi"):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x, attn_bias, raw=raw, v_src=v_src, skip_src=skip_src)
            else:
                x = layer(x, attn_bias)
        return self.final_ln(x)
    


class GlobalHyperNodeTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers=1,
        heads=8,
        dropout_rate=0.1, 
        edge_index=None,
        hops=5, 
        sparse_h= None,
        hg = None,
        args = None
    ):
        
    
        super().__init__()
        self.sparse_h = sparse_h    
        self.hg = hg
        self.args = args
        self.node_input_proj = nn.Linear( in_channels , hidden_channels)

        mlp1_layers = args.w1
        mlp2_layers = args.w3
        mlp3_layers = args.w5
        tr1_layers = args.w2
        tr2_layers = args.w4
        
        self.k = args.mp_steps# for k -1  hop neighbours

        classifier_layers = args.clf_layer

        self.rwpe = args.use_rwpe
        self.lappe = args.use_lappe
        
        self.pre_transform = args.pre_transform

        self.gate_exp = args.use_avg_gate

        self.subhg = args.extract_subgraph
        self.theory_encoder = getattr(args, "theory_encoder", False)
        self.feat_mode = getattr(args, "feat_mode", "real")
        self.const_no_dropout = getattr(args, "const_no_dropout", False)
        assert self.feat_mode in ("real", "ones")
        variant = getattr(args, "variant", None)
        if variant is None:
            self.v_src = getattr(args, "v_src", "phi")
            self.skip_src = getattr(args, "skip_src", "phi")
        else:
            variants = {
                "full": ("phi", "phi"),
                "qk_leak": ("raw", "phi"),
                "qk_clean": ("raw", "raw"),
                "qkv_clean": ("phi", "raw"),
            }
            assert variant in variants, f"unknown variant: {variant}"
            self.v_src, self.skip_src = variants[variant]
        assert self.v_src in ("phi", "raw") and self.skip_src in ("phi", "raw")
        if (self.v_src, self.skip_src) != ("phi", "phi"):
            assert n_layers == 1, "non-default structural variants require n_layers == 1"
            assert self.k > 1, "mp_steps == 1 skips the structure branch; the ablation would be meaningless"
            assert not (self.rwpe or self.lappe), "turn PEs off for this ablation"
        assert not (getattr(args, "mixer", "attn") == "mlp" and (self.v_src, self.skip_src) != ("phi", "phi")), "mlp mixer uses the full variant"
        if mlp1_layers > 0:
            self.W1 = MLP(in_channels=hidden_channels, out_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=mlp1_layers,
                dropout=dropout_rate, Normalization='ln', InputNorm=True)
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(in_channels=hidden_channels, out_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=mlp2_layers,
                dropout=dropout_rate, Normalization='ln', InputNorm=True)
        else:
            self.W2 = nn.Identity()

        if mlp3_layers > 0:
            self.W3 = MLP(in_channels=hidden_channels, out_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=mlp3_layers,
                dropout=dropout_rate, Normalization='ln', InputNorm=True)
        else:
            self.W3 = nn.Identity()

        if tr1_layers > 0:
            self.W_tr1 = MLP(in_channels=hidden_channels, out_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=tr1_layers,
                dropout=dropout_rate, Normalization='ln', InputNorm=True)
        else:
            self.W_tr1 = nn.Identity()
        if tr2_layers > 0:
            self.W_tr2 = MLP(in_channels=hidden_channels, out_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=tr2_layers,
                dropout=dropout_rate, Normalization='ln', InputNorm=True)
        else:
            self.W_tr2 = nn.Identity()

        self.classifier = MLP(in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=classifier_layers,
            dropout=dropout_rate,
            Normalization='ln',
            InputNorm=False)
       
        if self.rwpe:
            rw_pe = hypergraph.hypergraph_rw_pe_multiple(sparse_h, t=32, rw_type= "WE")
            self.register_buffer("rw_pe", torch.tensor(rw_pe, dtype=torch.float32))
            self.rw_proj = nn.Linear(32, hidden_channels)
            self.gate_rwpe   = nn.Linear(hidden_channels, hidden_channels)
        if self.lappe:
            lap_pe = hypergraph.hypergraph_norm_laplacian_pe(sparse_h, k = 32)
            self.register_buffer("lap_pe", torch.tensor(lap_pe, dtype=torch.float32))
            self.lap_proj = nn.Linear(32, hidden_channels)
            self.gate_lap  = nn.Linear(hidden_channels, hidden_channels)
        
        
        
        
        self.dropout = nn.Dropout(dropout_rate)

   
        self.shared_transformer = BasicTransformer(
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            heads=heads,
            dropout_rate=dropout_rate,
            mixer=getattr(args, "mixer", "attn")
        )
        
        if self.subhg:  
            if getattr(args, "subhg_encoder", "v1_sum") == "v1_sum":
                # paper numbering W1..W4 = self.W1, self.W_tr1, self.W2, self.W_tr2 (same mapping as the V1 loop)
                self.se_extractor = SubHypergraphV1Extractor(self.W1, self.W_tr1, self.W2, self.W_tr2,
                                                             num_layers=args.mp_steps_subhg)
            else:
                self.se_extractor = SubHypergraphStructureExtractor(embed_dim=hidden_channels,
                                                                    num_layers=args.mp_steps_subhg,
                                                                    dropout=dropout_rate)




        self.gate_node_emb = nn.Linear(hidden_channels, hidden_channels)
        self.gate_se   = nn.Linear(hidden_channels, hidden_channels)
        self.perturb = None   # set only by gate_perturbation.py
       


        
    def reset_parameters(self):
        self.node_input_proj.reset_parameters()
        
        self.classifier.reset_parameters()
        if isinstance(self.W3, MLP):
            self.W3.reset_parameters()

        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W_tr1, MLP):
            self.W_tr1.reset_parameters()
        if isinstance(self.W_tr2, MLP):
            self.W_tr2.reset_parameters()

        self.shared_transformer.reset_parameters()
        if self.subhg:
            self.se_extractor.reset_parameters()
        if self.rwpe:
            self.rw_proj.reset_parameters()
        if self.lappe:
            self.lap_proj.reset_parameters()
      
        
        if self.lappe:
            self.gate_lap.reset_parameters()
        if self.rwpe:
            self.gate_rwpe.reset_parameters()
        self.gate_node_emb.reset_parameters()
        self.gate_se.reset_parameters()
        
        
    def _apply_perturb(self, x_, Xv):
        """Gate-perturbation test: give node p['node'] another node's structure and/or features."""
        p = self.perturb
        v = p["node"]
        if p.get("struct_from") is not None:
            Xv = Xv.clone()
            Xv[v] = Xv[p["struct_from"]]
        if p.get("feat_from") is not None:
            x_ = x_.clone()
            x_[v] = x_[p["feat_from"]]
        return x_, Xv

    def forward(self, data):
   
        

        if self.lappe:
            lap_pe = self.lap_pe  # [N, k]

            if self.training:
                sign_flip = torch.randint(
                    0, 2, (lap_pe.size(1),), device=lap_pe.device
                ).float()
                sign_flip = sign_flip * 2 - 1  
                lap_pe = lap_pe * sign_flip.unsqueeze(0)

            lap_pe = self.lap_proj(lap_pe)
            g_lap = torch.sigmoid(self.gate_lap(lap_pe))

        if self.rwpe:
            rw_pe = self.rw_proj(self.rw_pe)
            g_rwpe   = torch.sigmoid(self.gate_rwpe(rw_pe))


        x = data.x
        if self.feat_mode == "ones":
            x = torch.ones_like(data.x)

        # x = torch.randn_like(x) 
        if not (self.feat_mode == "ones" and self.const_no_dropout):
            x = self.dropout(x)
        x = F.relu(self.node_input_proj(x))
        x0 = x
        V, E = data.edge_index[0], data.edge_index[1]
        N = x.shape[-2]
        if not (self.feat_mode == "ones" and self.const_no_dropout):
            x = self.dropout(x)
        x_ = x
        Xv_=[]
       


        if self.subhg:
           
            subhg_features = self.se_extractor(x_, data.sub_E, data.sub_root, data.sub_node_index)
            Xv= subhg_features

        elif self.k == 1:
            x=x_
    
        else:

            
            ones = torch.ones_like(E, dtype=torch.float32)
            edge_degree = torch_scatter.scatter(ones, E, dim=-1, reduce="add")
            node_degree = torch_scatter.scatter(ones, V, dim=-1, reduce="add", dim_size=N)
            edge_degree = edge_degree.unsqueeze(-1).clamp(min=1) 
            node_degree = node_degree.unsqueeze(-1).clamp(min=1)

            for _ in range(self.k - 1):
                

                Xve = self.W1(x)[..., V, :] # [nnz, C]
                Xe_sum = torch_scatter.scatter(Xve, E, dim=-2, reduce="add") # [E, C]

                if self.theory_encoder:
                    Xe = Xe_sum
                else:
                    Xe_mlp = self.W_tr1(Xe_sum)
                    Xe = Xe_mlp / edge_degree

                Xe = self.W2(Xe)
               
                Xev = Xe[..., E, :] # [nnz, C]
               
                Xv_sum = torch_scatter.scatter(Xev, V, dim=-2, reduce="add", dim_size=N) # [N, C]

                if self.theory_encoder:
                    Xv = Xv_sum
                else:
                    Xv_mlp = self.W_tr2(Xv_sum)
                    Xv = Xv_mlp / node_degree
                
                Xv_.append(Xv)
                x = Xv

            Xv = torch.mean(torch.stack(Xv_, dim=0), dim=0)

        if self.theory_encoder:
            Xv = torch.log1p(Xv.abs()) * torch.sign(Xv)

        if self.subhg or self.k > 1:
            self.saved_Xv = Xv.detach()
            self.saved_h0 = x_.detach()
            if self.perturb is not None and not self.perturb.get("fixed_gate", False):
                x_, Xv = self._apply_perturb(x_, Xv)   # gates are recomputed from the swapped inputs
       
       
        g_proc = torch.sigmoid(self.gate_node_emb(x_))
        if self.k > 1:
            g_se   = torch.sigmoid(self.gate_se(Xv))
        if self.perturb is not None and self.perturb.get("fixed_gate", False) and (self.subhg or self.k > 1):
            x_, Xv = self._apply_perturb(x_, Xv)       # original gates kept (secondary test)
        if self.k ==1:
            x=g_proc*x_
        else:
            x = g_proc*x_ + g_se*Xv
        

        if not self.subhg and self.k > 1:
            self.saved_node_degree = node_degree.squeeze(-1).detach().cpu().numpy()
            self.saved_g_proc = g_proc.mean(dim=-1).detach().cpu().numpy()
            self.saved_g_se = g_se.mean(dim=-1).detach().cpu().numpy()

      

        if self.gate_exp:
            x = 0.5*x_ + 0.5*Xv
       
        if self.rwpe:
            x = x + g_rwpe*rw_pe
        if self.lappe:
            x = x + g_lap*lap_pe

       
        raw = x_
        abl = dict(v_src=self.v_src, skip_src=self.skip_src)

        if self.pre_transform:
            x = self.dropout(F.relu(self.W3(x)))
            if (self.v_src, self.skip_src) != ("phi", "phi"):
                raw = self.dropout(F.relu(self.W3(raw)))
            x = self.shared_transformer(x.unsqueeze(0), attn_bias=None,
                                        raw=raw.unsqueeze(0), **abl).squeeze(0)
        else:
            x = self.shared_transformer(x.unsqueeze(0), attn_bias=None,
                                        raw=raw.unsqueeze(0), **abl).squeeze(0)
            x = self.dropout(F.relu(self.W3(x)))
                
        node_emb = x
        
      
        return self.classifier(node_emb)                  # [N, C]
       
       

    

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, mixer="attn"):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        # self.self_attention_norm = nn.BatchNorm1d(hidden_size)
        self.mixer = mixer
        if mixer == "mlp":
            # node-wise MLP, no cross-node mixing: 4d^2+3d params vs 4d^2+4d for attention
            self.self_attention = FeedForwardNetwork(hidden_size, 2 * hidden_size, dropout_rate)
        else:
            self.self_attention = MultiHeadAttention(
                hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        # self.ffn_norm = nn.BatchNorm1d(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def reset_parameters(self):
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x, attn_bias=None, raw=None, v_src="phi", skip_src="phi"):
        y = self.self_attention_norm(x)
        v_in = self.self_attention_norm(raw) if v_src == "raw" else y
        y = self.self_attention(y) if self.mixer == "mlp" else self.self_attention(y, y, v_in, attn_bias)
        y = self.self_attention_dropout(y)
        x = (raw if skip_src == "raw" else x) + y
        # x=y
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


    


# class MultiHeadAttention(nn.Module):
#     def __init__(self, hidden_size, attention_dropout_rate, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         self.att_size = hidden_size // num_heads

#         self.linear_q = nn.Linear(hidden_size, hidden_size)
#         self.linear_k = nn.Linear(hidden_size, hidden_size)
#         self.linear_v = nn.Linear(hidden_size, hidden_size)
#         self.output_layer = nn.Linear(hidden_size, hidden_size)

#         self.dropout = attention_dropout_rate

#     def reset_parameters(self):
#         self.linear_q.reset_parameters()
#         self.linear_k.reset_parameters()
#         self.linear_v.reset_parameters()
#         self.output_layer.reset_parameters()

#     def forward(self, q, k, v, attn_bias=None):
#         B, L, _ = q.shape
#         H = self.num_heads
#         d = self.att_size

#         q = self.linear_q(q).view(B, L, H, d).transpose(1, 2)
#         k = self.linear_k(k).view(B, L, H, d).transpose(1, 2)
#         v = self.linear_v(v).view(B, L, H, d).transpose(1, 2)
#         # q= q /temp
       

#         # attn_mask=attn_bias.unsqueeze(0).unsqueeze(0) if attn_bias is not None else None
#         # THIS LINE ENABLES FLASH ATTENTION
#         out = F.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=attn_bias.unsqueeze(0).unsqueeze(0) if attn_bias is not None else None,
#             dropout_p=self.dropout if self.training else 0.0,
#             is_causal=False
#         )

#         out = out.transpose(1, 2).contiguous().view(B, L, -1)
#         return self.output_layer(out)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_dropout_rate)

    def reset_parameters(self):
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.out_proj.reset_parameters()

    def feature_map(self, x):
        # positive feature map for linear attention
        return torch.relu(x) + 1e-6

    def forward(self, q, k, v, attn_bias=None):
        B, L, D = q.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(q).view(B, L, H, d).transpose(1, 2)  # [B,H,L,d]
        k = self.k_proj(k).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(v).view(B, L, H, d).transpose(1, 2)

        q = self.feature_map(q)
        k = self.feature_map(k)

        # Compute KV summary: [B,H,d,d]
        kv = torch.einsum("bhld,bhlm->bhdm", k, v)

        # Compute normalization term: [B,H,L,1]
        z = 1 / (torch.einsum("bhld,bhd->bhl", q, k.sum(dim=2)) + 1e-6)
        z = z.unsqueeze(-1)

        # Final output: [B,H,L,d]
        out = torch.einsum("bhld,bhdm->bhlm", q, kv)
        out = out * z

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
    

# class SubHypergraphStructureExtractor(nn.Module):
#     def __init__(self, embed_dim, num_layers=1, dropout=0.0):
#         super().__init__()
#         self.num_layers = 2
#         self.embed_dim = embed_dim

#         mlp1_layers = 0
#         mlp2_layers = 0
    
#         if mlp1_layers > 0:
#             self.W1 = MLP(in_channels=embed_dim, out_channels=embed_dim, hidden_channels=embed_dim, num_layers=mlp1_layers,
#                 dropout=dropout, Normalization='ln', InputNorm=True)
#         else:
#             self.W1 = nn.Identity()

#         if mlp2_layers > 0:
#             self.W2 = MLP(in_channels=embed_dim, out_channels=embed_dim, hidden_channels=embed_dim, num_layers=mlp2_layers,
#                 dropout=dropout, Normalization='ln', InputNorm=True)
#         else:
#             self.W2 = nn.Identity()
#         self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

#     def reset_parameters(self):
#         if isinstance(self.W1, MLP):
#             self.W1.reset_parameters()
#         if isinstance(self.W2, MLP):
#             self.W2.reset_parameters()


#     def forward(self, x, sub_E, sub_root, sub_node_index):
#         N = x.shape[0]

#         # --- Precompute all indices once ---
#         max_local_E = sub_E.max().item() + 1
#         global_E = sub_root * max_local_E + sub_E          # [nnz]
#         global_V = sub_root * N + sub_node_index           # [nnz]

#         # Unique (root, node) pairs — computed once, reused every layer
#         unique_global_V, inverse_indices = torch.unique(
#             global_V, return_inverse=True
#         )                                                  # [M], [nnz]
#         unique_roots = unique_global_V // N                # [M]

#         # --- Initial node features ---
#         h = x[sub_node_index]                              # [nnz, C]
#         h_unique_hops = []
#         for layer_idx in range(self.num_layers):
            
#             # --- Node -> Hyperedge ---
#             Xve = self.W1(h)                                    # [nnz, C]
#             Xe = torch_scatter.scatter(
#                 Xve, global_E, dim=0, reduce="mean"
#             )                                              # [num_global_E, C]

#             # --- Hyperedge -> Node ---
           
#             Xev = self.W2(Xe[global_E]) # [nnz, C]
           

#             # --- Aggregate to unique (root, node) pairs ---
#             # This is the standard HGNN hyperedge->node aggregation step:
#             # h_v = mean over all hyperedges e containing v of W2([h_v, h_e])
#             # inverse_indices maps each incidence pair to its unique (root,node) slot
#             h_unique = torch_scatter.scatter(
#                 Xev, inverse_indices,
#                 dim=0, reduce="mean",
#                 dim_size=unique_global_V.shape[0]
#             )                                              # [M, C]
#             h_unique_hops.append(h_unique)
#             # Broadcast back to incidence pairs for next layer
#             h = h_unique[inverse_indices]                  # [nnz, C]
#         h_unique_accumulated = torch.mean(torch.stack(h_unique_hops, dim=0), dim=0)
#         # --- Pool unique nodes -> root (sum preserves topological volume) ---
#         x_sub = torch_scatter.scatter(
#             h_unique_accumulated, unique_roots,
#             dim=0, reduce="mean",
#             dim_size=N
#         )                                                  # [N, C]

#         # --- Fuse with original features ---
#         x_struct = self.out_proj(torch.cat([x, x_sub], dim=-1))  # [N, C]
#         return x_struct

class SubHypergraphRootExtractor(nn.Module):  # root readout
    def __init__(self, embed_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.num_layers = 3
        self.embed_dim = embed_dim

        mlp1_layers = 0
        mlp2_layers = 0
        tr1_layers = 0
        tr2_layers = 0


        if mlp1_layers > 0:
            self.W1 = MLP(in_channels=embed_dim, out_channels=embed_dim, hidden_channels=embed_dim, num_layers=mlp1_layers,
                dropout=dropout, Normalization='ln', InputNorm=True)
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(in_channels=embed_dim, out_channels=embed_dim, hidden_channels=embed_dim, num_layers=mlp2_layers,
                dropout=dropout, Normalization='ln', InputNorm=True)
        else:
            self.W2 = nn.Identity()

        if tr1_layers > 0:
            self.W_tr1 = MLP(in_channels=embed_dim, out_channels=embed_dim, hidden_channels=embed_dim, num_layers=tr1_layers,
                dropout=dropout, Normalization='ln', InputNorm=True)
        else:
            self.W_tr1 = nn.Identity()
        if tr2_layers > 0:
            self.W_tr2 = MLP(in_channels=embed_dim, out_channels=embed_dim  , hidden_channels=embed_dim, num_layers=tr2_layers,
                dropout=dropout, Normalization='ln', InputNorm=True)
        else:
            self.W_tr2 = nn.Identity()
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W_tr1, MLP):
            self.W_tr1.reset_parameters()
        if isinstance(self.W_tr2, MLP):
            self.W_tr2.reset_parameters()

    def forward(self, x, sub_E, sub_root, sub_node_index):
        N = x.shape[0]

        # --- Precompute all indices once ---
        max_local_E = sub_E.max().item() + 1
        global_E = sub_root * max_local_E + sub_E          # [nnz]
        global_V = sub_root * N + sub_node_index           # [nnz]

        # Unique (root, node) pairs — computed once, reused every layer
        unique_global_V, inverse_indices = torch.unique(
            global_V, return_inverse=True
        )                                                  # [M], [nnz]
        unique_roots = unique_global_V // N                # [M]

        # ==========================================
        # NEW: Precompute degrees for normalization
        # ==========================================
        ones = torch.ones_like(global_E, dtype=x.dtype)

        # Edge Degree: How many incidence pairs map to each global hyperedge?
        edge_degree = torch_scatter.scatter(
            ones, global_E, dim=0, reduce="add"
        ).unsqueeze(-1).clamp(min=1)                       # [num_global_E, 1]

        # Node Degree: How many incidence pairs map to each unique (root, node)?
        node_degree = torch_scatter.scatter(
            ones, inverse_indices, dim=0, reduce="add", 
            dim_size=unique_global_V.shape[0]
        ).unsqueeze(-1).clamp(min=1)                       # [M, 1]
        # ==========================================

        # --- Initial node features ---
        h = x[sub_node_index]                              # [nnz, C]
        h_unique_hops = []
        for layer_idx in range(self.num_layers):
            
            # --- Node -> Hyperedge ---
            Xve = self.W1(h)                                    # [nnz, C]
            Xe_sum = torch_scatter.scatter(
                Xve, global_E, dim=0, reduce="add"
            )                                              # [num_global_E, C]

            Xe = self.W_tr1(Xe_sum) / edge_degree              # Normalize by edge degree
            # --- Hyperedge -> Node ---
            
            Xev = self.W2(Xe[global_E]) # [nnz, C]
            

            # --- Aggregate to unique (root, node) pairs ---
            # This is the standard HGNN hyperedge->node aggregation step:
            # h_v = mean over all hyperedges e containing v of W2([h_v, h_e])
            # inverse_indices maps each incidence pair to its unique (root,node) slot
            h_unique_sum = torch_scatter.scatter(
                Xev, inverse_indices,
                dim=0, reduce="add",
                dim_size=unique_global_V.shape[0]
            )                                              # [M, C]
            
            h_unique = self.W_tr2(h_unique_sum) / node_degree # Normalize by node degree
            h_unique_hops.append(h_unique)
            # Broadcast back to incidence pairs for next layer
            h = h_unique[inverse_indices]                  # [nnz, C]
        h_unique_accumulated = torch.mean(torch.stack(h_unique_hops, dim=0), dim=0)
        
        # --- Target Node Readout (Experiment) ---
        # 1. Isolate the indices where the sub-node is exactly the root node
        root_mask = (sub_node_index == sub_root)
        root_inverse_indices = inverse_indices[root_mask]
        
        # 2. Extract the unique mapped IDs for the root nodes
        # Since torch.unique returns sorted items, and the roots go from 0 to N-1,
        # target_indices will be perfectly ordered [0, 1, ..., N-1]
        target_indices = torch.unique(root_inverse_indices)

        # 3. Readout just the target nodes from the accumulated sub-hypergraph states
        x_sub_target = h_unique_accumulated[target_indices] # [N, C]

        # --- Fuse with original features ---
        # x_struct = self.out_proj(torch.cat([x, x_sub_target], dim=-1))  # [N, C]
        return x_sub_target


class SubHypergraphStructureExtractor(nn.Module):
    def __init__(self, embed_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.W1_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False) 
            for _ in range(num_layers)
        ])
        # W2 now takes concat of current node state + hyperedge message
        self.W2_layers = nn.ModuleList([
            nn.Linear(embed_dim * 2, embed_dim, bias=False) 
            for _ in range(num_layers)
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def reset_parameters(self):
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x, sub_E, sub_root, sub_node_index):
        N = x.shape[0]

        # --- Precompute all indices once ---
        max_local_E = sub_E.max().item() + 1
        global_E = sub_root * max_local_E + sub_E          # [nnz]
        global_V = sub_root * N + sub_node_index           # [nnz]

        # Unique (root, node) pairs — computed once, reused every layer
        unique_global_V, inverse_indices = torch.unique(
            global_V, return_inverse=True
        )                                                  # [M], [nnz]
        unique_roots = unique_global_V // N                # [M]

        # --- Initial node features ---
        h = x[sub_node_index]                              # [nnz, C]

        for layer_idx in range(self.num_layers):
            W1 = self.W1_layers[layer_idx]
            W2 = self.W2_layers[layer_idx]

            # --- Node -> Hyperedge ---
            Xve = W1(h)                                    # [nnz, C]
            Xe = torch_scatter.scatter(
                Xve, global_E, dim=0, reduce="mean"
            )                                              # [num_global_E, C]

            # --- Hyperedge -> Node ---
            # Concat current node state with incoming hyperedge message
            # so each node update is informed by both its current state
            # and the aggregated context of the hyperedge it belongs to
            Xev = W2(torch.cat([h, Xe[global_E]], dim=-1)) # [nnz, C]
            Xev = self.relu(Xev)
            Xev = self.dropout(Xev)

            # --- Aggregate to unique (root, node) pairs ---
            # This is the standard HGNN hyperedge->node aggregation step:
            # h_v = mean over all hyperedges e containing v of W2([h_v, h_e])
            # inverse_indices maps each incidence pair to its unique (root,node) slot
            h_unique = torch_scatter.scatter(
                Xev, inverse_indices,
                dim=0, reduce="mean",
                dim_size=unique_global_V.shape[0]
            )                                              # [M, C]

            # Broadcast back to incidence pairs for next layer
            h = h_unique[inverse_indices]                  # [nnz, C]

        # --- Pool unique nodes -> root (sum preserves topological volume) ---
        x_sub = torch_scatter.scatter(
            h_unique, unique_roots,
            dim=0, reduce="add",
            dim_size=N
        )                                                  # [N, C]

        # --- Fuse with original features ---
        x_struct = self.out_proj(torch.cat([x, x_sub], dim=-1))  # [N, C]
        return x_struct


class SubHypergraphV1Extractor(nn.Module):
    """V2 as described in Sec. 4.1: L layers of the V1 scatter-gather update (Eqs. 10-11) inside each
    K-hop sub-hypergraph S_v, then Set-Union Pooling h_v = sum_{u in V_v} hbar_vu (no output projection).
    W1-W4 (paper numbering) are the same modules the V1 encoder uses; they are owned by the parent model."""

    def __init__(self, W1, W2, W3, W4, num_layers):
        super().__init__()
        self.W1, self.W2, self.W3, self.W4 = W1, W2, W3, W4
        self.num_layers = num_layers

    def reset_parameters(self):
        pass  # W1-W4 are reset by GlobalHyperNodeTransformer

    def forward(self, x, sub_E, sub_root, sub_node_index):
        N = x.shape[0]
        max_local_E = sub_E.max().item() + 1
        global_E = sub_root * max_local_E + sub_E
        global_V = sub_root * N + sub_node_index
        unique_global_V, inverse_indices = torch.unique(global_V, return_inverse=True)
        unique_roots = unique_global_V // N
        M = unique_global_V.shape[0]
        ones = torch.ones_like(global_E, dtype=x.dtype)
        edge_degree = torch_scatter.scatter(ones, global_E, dim=0, reduce="add").unsqueeze(-1).clamp(min=1)
        node_degree = torch_scatter.scatter(ones, inverse_indices, dim=0, reduce="add",
                                            dim_size=M).unsqueeze(-1).clamp(min=1)
        h = x[unique_global_V % N]                                   # one row per (root, u)
        for _ in range(self.num_layers):
            he = torch_scatter.scatter(self.W1(h)[inverse_indices], global_E, dim=0, reduce="add")
            he = self.W3(self.W2(he) / edge_degree)                  # Eq. (10)
            hv = torch_scatter.scatter(he[global_E], inverse_indices, dim=0, reduce="add", dim_size=M)
            h = self.W4(hv) / node_degree                            # Eq. (11)
        return torch_scatter.scatter(h, unique_roots, dim=0, reduce="add", dim_size=N)   # Set-Union Pooling