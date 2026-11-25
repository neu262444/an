from types import SimpleNamespace

import torch

from models.HSAT import GlobalHyperNodeTransformer


def _args(kernel=None):
    values = dict(
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
        feat_mode="real",
        const_no_dropout=False,
        mixer="attn",
    )
    if kernel is not None:
        values["kernel"] = kernel
    return SimpleNamespace(**values)


def _model_and_output(data, kernel=None):
    torch.manual_seed(42)
    model = GlobalHyperNodeTransformer(
        in_channels=data.x.size(-1),
        hidden_channels=4,
        out_channels=3,
        n_layers=1,
        heads=2,
        dropout_rate=0.0,
        args=_args(kernel),
    )
    model.eval()
    with torch.no_grad():
        return model(data)


def test_exp_and_relu_attention_kernels_preserve_output_shape():
    torch.manual_seed(7)
    data = SimpleNamespace(
        x=torch.randn(6, 3),
        edge_index=torch.stack(
            (torch.randint(0, 6, (12,)), torch.arange(12) % 3)
        ),
    )

    assert _model_and_output(data, "exp").shape == (6, 3)
    assert _model_and_output(data, "relu").shape == (6, 3)


def test_explicit_relu_matches_kernel_default():
    torch.manual_seed(7)
    data = SimpleNamespace(
        x=torch.randn(6, 3),
        edge_index=torch.stack(
            (torch.randint(0, 6, (12,)), torch.arange(12) % 3)
        ),
    )

    output_without_flag = _model_and_output(data)
    output_with_relu = _model_and_output(data, "relu")

    assert torch.equal(output_without_flag, output_with_relu)
