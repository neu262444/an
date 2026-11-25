import ast
from types import SimpleNamespace

import torch
import torch.nn as nn

from datasets import extract_sub_hypergraphs
from models.HSAT import (
    GlobalHyperNodeTransformer,
    SubHypergraphV1Extractor,
)


def test_train_parser_subhg_encoder_default_is_v1_sum():
    source = open("train.py", "r", encoding="utf-8").read()
    tree = ast.parse(source)
    found = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == "--subhg_encoder":
            for keyword in node.keywords:
                if keyword.arg == "default":
                    found = ast.literal_eval(keyword.value)
    assert found == "v1_sum"


def test_missing_subhg_encoder_args_select_v1_sum_extractor():
    args = SimpleNamespace(
        w1=0,
        w2=0,
        w3=0,
        w4=0,
        w5=0,
        mp_steps=2,
        mp_steps_subhg=2,
        clf_layer=1,
        use_rwpe=False,
        use_lappe=False,
        pre_transform=True,
        use_avg_gate=False,
        extract_subgraph=True,
        mixer="attn",
    )
    edge_index = torch.tensor([[0, 1], [0, 0]])
    model = GlobalHyperNodeTransformer(
        in_channels=1,
        hidden_channels=4,
        out_channels=2,
        n_layers=1,
        heads=2,
        dropout_rate=0.0,
        edge_index=edge_index,
        args=args,
    )
    assert isinstance(model.se_extractor, SubHypergraphV1Extractor)


def test_v2_v1_sum_appendix_b():
    hg1 = [[0, 1, 2], [1, 3, 4], [4, 5, 6], [2, 7, 6]]
    hg2 = [[8, 9, 10], [9, 10, 11], [12, 13, 14], [13, 14, 15]]
    edges = hg1 + hg2
    V = torch.tensor([v for edge in edges for v in edge])
    E = torch.tensor([j for j, edge in enumerate(edges) for _ in edge])
    sub = extract_sub_hypergraphs(torch.stack([V, E]), 16, k_hop=2)
    x = torch.ones(16, 1)
    extractor = SubHypergraphV1Extractor(
        nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), num_layers=2
    )
    output = extractor(x, *sub)
    assert torch.allclose(output[0], torch.tensor([7.0]))
    assert torch.allclose(output[8], torch.tensor([4.0]))
