from types import SimpleNamespace
from pathlib import Path
import argparse
import sys

import torch
import torch_geometric

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets
from models.HSAT import GlobalHyperNodeTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theory_encoder", action="store_true")
    cli_args = parser.parse_args()

    torch.manual_seed(42)
    args = SimpleNamespace(
        cuda=0,
        dname="cora",
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
        feat_mode="ones",
        const_no_dropout=True,
        theory_encoder=cli_args.theory_encoder,
    )
    transform = torch_geometric.transforms.Compose(
        [datasets.AddHypergraphSelfLoops()]
    )
    data = datasets.HypergraphDataset(
        root="data/cora",
        name="cora",
        feature_noise="1",
        transform=transform,
        args=args,
    )._data

    print(f"{'variant':<10} {'attn std max':>14} {'logits std max':>16}")
    variants = ("qk_clean", "qkv_clean", "qk_leak", "full")
    for variant in variants:
        torch.manual_seed(42)
        args.variant = variant
        model = GlobalHyperNodeTransformer(
            in_channels=data.num_features,
            hidden_channels=64,
            out_channels=data.num_classes,
            n_layers=1,
            heads=8,
            dropout_rate=0.5,
            args=args,
        )
        model.eval()

        attention_outputs = []
        handle = model.shared_transformer.layers[0].self_attention.register_forward_hook(
            lambda module, inputs, output: attention_outputs.append(output.detach())
        )
        with torch.no_grad():
            logits = model(data)
        handle.remove()

        attention_std = attention_outputs[0].squeeze(0).std(dim=0).max().item()
        logits_std = logits.std(dim=0).max().item()
        xv_std = model.saved_Xv.std(dim=0).max().item()
        print(f"{variant:<10} {attention_std:14.8g} {logits_std:16.8g}")

    print(f"Xv.std(dim=0).max(): {xv_std:.8g}")


if __name__ == "__main__":
    main()
