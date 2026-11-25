import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight.shape[1])
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, sparse_input):
        return torch.sparse.mm(sparse_input, self.weight.t()) + self.bias


def create_projection_matrix(num_rows, dimension, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed) % (2**31 - 1))
    blocks = []
    for _ in range(num_rows // dimension):
        matrix = torch.randn(dimension, dimension, generator=generator)
        blocks.append(torch.linalg.qr(matrix).Q.t())
    remainder = num_rows % dimension
    if remainder:
        matrix = torch.randn(dimension, dimension, generator=generator)
        blocks.append(torch.linalg.qr(matrix).Q.t()[:remainder])
    projection = torch.cat(blocks, dim=0)
    multiplier = torch.linalg.vector_norm(
        torch.randn(num_rows, dimension, generator=generator), dim=1
    )
    return projection * multiplier.unsqueeze(1)


def softmax_kernel(values, is_query, projection_matrix):
    values = values / math.sqrt(math.sqrt(values.shape[-1]))
    projection_matrix = projection_matrix.to(values.device, values.dtype)
    projected = torch.einsum("bnhd,md->bnhm", values, projection_matrix)
    squared_norm = values.square().sum(dim=-1, keepdim=True) / 2
    if is_query:
        maximum = projected.max(dim=-1, keepdim=True).values
    else:
        maximum = projected.amax(dim=(-2, -1), keepdim=True)
    return torch.exp(projected - squared_norm - maximum) / math.sqrt(projection_matrix.shape[0]) + 1e-6


def linear_attention(query, key, value, projection_matrix):
    query_features = softmax_kernel(query, True, projection_matrix).permute(1, 0, 2, 3)
    key_features = softmax_kernel(key, False, projection_matrix).permute(1, 0, 2, 3)
    value = value.permute(1, 0, 2, 3)

    key_value = torch.einsum("nbhm,nbhd->bhmd", key_features, value)
    numerator = torch.einsum("nbhm,bhmd->nbhd", query_features, key_value)
    key_sum = key_features.sum(dim=0)
    denominator = torch.einsum("nbhm,bhm->nbh", query_features, key_sum)
    output = numerator / denominator.clamp_min(1e-6).unsqueeze(-1)
    return output.permute(1, 0, 2, 3)


class HyperGTLayer(nn.Module):
    def __init__(self, hidden_channels, heads, random_features, dropout, rb_order, rb_trans):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.head_dim = hidden_channels // heads
        self.random_features = random_features
        self.rb_order = rb_order
        self.rb_trans = rb_trans
        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.key = nn.Linear(hidden_channels, hidden_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        if rb_order:
            self.relational_bias = nn.Parameter(torch.full((rb_order, heads), 0.1))

    def reset_parameters(self):
        self.query.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()
        self.output.reset_parameters()
        self.norm.reset_parameters()
        if self.rb_order:
            nn.init.constant_(self.relational_bias, 0.1)

    def relational_message(self, value, edge_index, node_count):
        row, column = edge_index
        source_degree = torch.bincount(column, minlength=node_count).float().clamp_min(1)
        target_degree = torch.bincount(row, minlength=node_count).float().clamp_min(1)
        scale = source_degree[column].rsqrt() * target_degree[row].rsqrt()
        messages = []
        for head in range(self.heads):
            weight = self.relational_bias[:, head].sigmoid()[-1] if self.rb_trans == 'sigmoid' else self.relational_bias[-1, head]
            weighted = value[:, :, head, :] * weight * scale.view(1, -1, 1)
            messages.append(torch.zeros_like(value[:, :, head, :]).index_add(1, row, weighted))
        return torch.stack(messages, dim=2)

    def forward(self, hidden, edge_index, tau, use_edge_loss):
        residual = hidden
        normalized = self.norm(hidden)
        batch_size, node_count, _ = normalized.shape
        query = self.query(normalized).view(batch_size, node_count, self.heads, self.head_dim)
        key = self.key(normalized).view(batch_size, node_count, self.heads, self.head_dim)
        value = self.value(normalized).view(batch_size, node_count, self.heads, self.head_dim)
        projection = create_projection_matrix(self.random_features, self.head_dim, 0).to(hidden.device)
        attended = linear_attention(query / math.sqrt(tau), key / math.sqrt(tau), value, projection)

        edge_loss = hidden.new_zeros(())
        if self.rb_order:
            attended = attended + self.relational_message(value, edge_index, node_count)
        if use_edge_loss:
            query_features = softmax_kernel(query, True, projection)
            key_features = softmax_kernel(key, False, projection)
            row, column = edge_index
            attention_numerator = (query_features[:, row] * key_features[:, column]).sum(dim=-1)
            normalization = torch.einsum('bnhm,bhm->bnh', query_features, key_features.sum(dim=1))
            attention_weight = attention_numerator / normalization[:, column].clamp_min(1e-6)
            edge_loss = attention_weight.clamp_min(1e-8).log().mean()
        attended = attended.reshape(batch_size, node_count, self.hidden_channels)
        return residual + self.dropout(self.output(attended)), edge_loss


class HyperGT(nn.Module):
    """Standalone HyperGT implementation using HSAT's ``forward(data)`` API."""

    def __init__(self, num_features, num_classes, args, num_hyperedges):
        super().__init__()
        self.node_count = None
        self.num_hyperedges = num_hyperedges
        hidden_channels = args.hypergt_hidden_channels
        self.input = nn.Linear(num_features, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)
        self.layers = nn.ModuleList([
            HyperGTLayer(
                hidden_channels,
                args.hypergt_heads,
                args.hypergt_random_features,
                args.dropout,
                args.hypergt_rb_order,
                args.hypergt_rb_trans,
            )
            for _ in range(args.hypergt_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = args.dropout
        self.use_bn = args.hypergt_use_bn
        self.use_residual = args.hypergt_use_residual
        self.use_act = args.hypergt_use_act
        self.use_jk = args.hypergt_use_jk
        self.pe = args.hypergt_pe
        self.edge_loss_enabled = args.hypergt_edge_loss or getattr(args, 'sloss', False)
        self.edge_encoder = SparseLinear(num_hyperedges, hidden_channels)
        self.node_encoder = SparseLinear(args.hypergt_num_nodes, hidden_channels)
        self.last_edge_losses = []

    def reset_parameters(self):
        self.input.reset_parameters()
        self.input_norm.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_encoder.reset_parameters()

    def build_inputs(self, data):
        node_index, hyperedge_index = data.edge_index.long()
        node_count = data.x.shape[0]
        edge_count = int(hyperedge_index.max().item()) + 1
        values = torch.ones(node_index.numel(), device=data.x.device)
        incidence = torch.sparse_coo_tensor(
            torch.stack((node_index, hyperedge_index)), values,
            (node_count, edge_count), device=data.x.device
        ).coalesce()
        edge_degree = torch.sparse.sum(incidence, dim=0).to_dense().clamp_min(1).unsqueeze(-1)
        edge_features = torch.sparse.mm(incidence.t(), data.x) / edge_degree
        tokens = torch.cat((data.x, edge_features), dim=0)
        token_count = node_count + edge_count
        edge_tokens = hyperedge_index + node_count
        bipartite = torch.cat((
            torch.stack((node_index, edge_tokens)),
            torch.stack((edge_tokens, node_index)),
        ), dim=1)
        loops = torch.arange(token_count, device=data.x.device).repeat(2, 1)
        return tokens, torch.cat((bipartite, loops), dim=1), incidence

    def add_positional_encodings(self, tokens, incidence):
        node_count = incidence.shape[0]
        if 'HEPE' in self.pe:
            node_pe = self.edge_encoder(incidence)
            tokens = tokens + torch.cat((node_pe, torch.zeros_like(tokens[node_count:])), dim=0)
        if 'HtEPE' in self.pe:
            edge_pe = self.node_encoder(incidence.t())
            tokens = tokens + torch.cat((torch.zeros_like(tokens[:node_count]), edge_pe), dim=0)
        return tokens

    def forward(self, data):
        tokens, adjacency, incidence = self.build_inputs(data)
        self.node_count = data.x.shape[0]
        hidden = self.input(tokens).unsqueeze(0)
        if self.pe != 'none':
            hidden = self.add_positional_encodings(hidden.squeeze(0), incidence).unsqueeze(0)
        if self.use_bn:
            hidden = self.input_norm(hidden)
        hidden = F.dropout(F.elu(hidden), p=self.dropout, training=self.training)
        self.last_edge_losses = []
        for layer in self.layers:
            updated, edge_loss = layer(hidden, adjacency, 0.25, self.edge_loss_enabled)
            hidden = updated if self.use_residual else updated - hidden
            if self.use_act:
                hidden = F.elu(hidden)
            self.last_edge_losses.append(edge_loss)
        logits = self.classifier(hidden.squeeze(0))
        return logits[:self.node_count]