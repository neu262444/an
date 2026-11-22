from types import SimpleNamespace

import torch

from models.HSAT import GlobalHyperNodeTransformer


def _args(variant):
    return SimpleNamespace(
        w1=0,
        w2=0,
        w3=0,
        w4=0,
        w5=0,
        mp_steps=2,
        clf_layer=1,
        use_rwpe=False,
        use_lappe=False,
        pre_transform=True,
        use_avg_gate=False,
        extract_subgraph=False,
        v_src="phi",
        skip_src="phi",
        variant=variant,
        feat_mode="real",
        const_no_dropout=False,
    )


def _model_and_reference(data, variant):
    torch.manual_seed(42)
    model = GlobalHyperNodeTransformer(
        in_channels=data.x.size(-1),
        hidden_channels=4,
        out_channels=2,
        n_layers=1,
        heads=2,
        dropout_rate=0.0,
        args=_args(variant),
    )
    model.eval()

    transformer_input = []
    handle = model.shared_transformer.register_forward_pre_hook(
        lambda module, inputs: transformer_input.append(inputs[0].detach().clone())
    )
    with torch.no_grad():
        logits = model(data)
    handle.remove()

    # Reference the pre-ablation transformer path: each layer receives only Phi.
    reference = transformer_input[0]
    with torch.no_grad():
        for layer in model.shared_transformer.layers:
            reference = layer(reference, attn_bias=None)
        reference = model.shared_transformer.final_ln(reference)
        reference_logits = model.classifier(reference.squeeze(0))
    return logits, reference_logits


def test_default_and_full_match_legacy_logits():
    torch.manual_seed(7)
    node_features = torch.randn(6, 3)
    edge_index = torch.stack(
        (
            torch.randint(0, 6, (12,)),
            torch.arange(12) % 3,
        )
    )
    data = SimpleNamespace(x=node_features, edge_index=edge_index)

    logits_without_variant, saved_reference = _model_and_reference(data, None)
    logits_full, full_reference = _model_and_reference(data, "full")

    assert torch.equal(logits_without_variant, saved_reference)
    assert torch.equal(logits_full, saved_reference)
    assert torch.equal(full_reference, saved_reference)


def test_non_default_variant_requires_one_layer():
    data = SimpleNamespace(x=torch.randn(2, 3), edge_index=torch.tensor([[0, 1], [0, 0]]))
    try:
        GlobalHyperNodeTransformer(
            in_channels=3,
            hidden_channels=4,
            out_channels=2,
            n_layers=2,
            heads=2,
            args=_args("qk_leak"),
        )
    except AssertionError as error:
        assert "n_layers == 1" in str(error)
    else:
        raise AssertionError("non-default variants must reject n_layers > 1")


def test_constant_features_without_dropout_have_zero_node_std():
    torch.manual_seed(10)
    data = SimpleNamespace(
        x=torch.randn(6, 3),
        edge_index=torch.stack((torch.randint(0, 6, (12,)), torch.arange(12) % 3)),
    )
    args = _args(None)
    args.feat_mode = "ones"
    args.const_no_dropout = True
    model = GlobalHyperNodeTransformer(
        in_channels=data.x.size(-1),
        hidden_channels=4,
        out_channels=2,
        n_layers=1,
        heads=2,
        dropout_rate=0.5,
        args=args,
    )
    model.train()

    projected = []
    handle = model.node_input_proj.register_forward_hook(
        lambda module, inputs, output: projected.append(output.detach().clone())
    )
    model(data)
    handle.remove()

    assert torch.equal(projected[0].std(dim=0), torch.zeros(4))
