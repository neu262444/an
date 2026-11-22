import torch
import torch.nn as nn


class SparseStructuralAttention(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, gamma):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.gamma = gamma

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.out_proj.reset_parameters()

    @staticmethod
    def feature_map(values):
        return torch.relu(values) + 1e-6

    @staticmethod
    def normalized_incidence(edge_index, num_nodes, num_hyperedges, device):
        node_index, hyperedge_index = edge_index
        values = torch.ones(node_index.numel(), device=device)
        node_degree = torch.zeros(num_nodes, device=device)
        hyperedge_degree = torch.zeros(num_hyperedges, device=device)
        node_degree.scatter_add_(0, node_index, values)
        hyperedge_degree.scatter_add_(0, hyperedge_index, values)

        node_scale = node_degree.clamp_min(1).rsqrt()
        hyperedge_scale = hyperedge_degree.clamp_min(1).reciprocal()
        values = values * node_scale[node_index] * hyperedge_scale[hyperedge_index].sqrt()
        return torch.sparse_coo_tensor(
            edge_index, values, (num_nodes, num_hyperedges), device=device
        ).coalesce()

    def forward(self, x, edge_index, num_hyperedges):
        batch_size, node_count, _ = x.shape
        q = self.q_proj(x).view(batch_size, node_count, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, node_count, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, node_count, self.heads, self.head_dim).transpose(1, 2)
        q = self.feature_map(q)
        k = self.feature_map(k)

        kv = torch.einsum("bhnd,bhnm->bhdm", k, v)
        normalizer = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).clamp_min(1e-6)
        linear_output = torch.einsum("bhnd,bhdm->bhnm", q, kv)
        linear_output = linear_output / normalizer.unsqueeze(-1)

        incidence = self.normalized_incidence(
            edge_index, node_count, num_hyperedges, x.device
        )
        structural_output = []
        for head_values in v.transpose(0, 1):
            edge_values = torch.sparse.mm(incidence.transpose(0, 1), head_values[0])
            node_values = torch.sparse.mm(incidence, edge_values)
            structural_output.append(node_values.unsqueeze(0))
        structural_output = torch.cat(structural_output, dim=0).view(
            batch_size, self.heads, node_count, self.head_dim
        )

        output = self.gamma * linear_output + (1.0 - self.gamma) * structural_output
        output = output.transpose(1, 2).contiguous().view(batch_size, node_count, self.hidden_dim)
        return self.out_proj(self.dropout(output))


class HGraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, gamma):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = SparseStructuralAttention(hidden_dim, heads, dropout, gamma)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.PReLU()

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.attention.reset_parameters()

    def forward(self, x, edge_index, num_hyperedges):
        attended = self.attention(self.norm(x), edge_index, num_hyperedges)
        return self.activation(x + self.dropout(attended))


class HGraphormer(nn.Module):
    def __init__(self, num_features, num_classes, args):
        super().__init__()
        self.input_proj = nn.Linear(num_features, args.hgraphormer_hidden_dim)
        self.layers = nn.ModuleList([
            HGraphormerLayer(
                args.hgraphormer_hidden_dim,
                args.hgraphormer_heads,
                args.dropout,
                args.hgraphormer_gamma,
            )
            for _ in range(args.hgraphormer_layers)
        ])
        self.classifier = nn.Linear(args.hgraphormer_hidden_dim, num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        x = self.dropout(torch.relu(self.input_proj(data.x)))
        x = x.unsqueeze(0)
        edge_index = data.edge_index.long()
        num_hyperedges = int(edge_index[1].max().item()) + 1
        for layer in self.layers:
            x = layer(x, edge_index, num_hyperedges)
        return self.classifier(x.squeeze(0))
