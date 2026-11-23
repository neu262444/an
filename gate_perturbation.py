#!/usr/bin/env python3
"""Gate-perturbation test (reviewers MjoW W2, L1Ch W2; AC point 4).

For each test node v of a trained HSGF-T model:
  structure swap : replace v's structural embedding h_v with a random donor's h_u. The gates are recomputed
                   from the swapped inputs (primary). A fixed-gate variant keeps v's original gates (secondary;
                   there the perturbation is proportional to the gate by construction).
  feature swap   : replace v's projected features h0_v with the donor's h0_u (mirror test for the feature gate).
Only node v is changed. drop = p(yhat_v) before - after, where yhat_v is the unperturbed prediction.
Question: does the structure gate (and the structural share) predict the drop, also after controlling for degree?

usage: python gate_perturbation.py --dname ModelNet40 --tag HSAT_attn [--donors 5] [--max_nodes 1000]
needs checkpoints/<dname>_<tag>_run<k>.pt written by `train.py ... --save_ckpt`
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from scipy import stats

import datasets
import features
from models import GlobalHyperNodeTransformer


def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing the rank-linear effect of z."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    if np.std(rz) == 0:
        return spearman(x, y)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def deciles(g):
    return ((stats.rankdata(g, method="ordinal") - 1) * 10 // len(g)).astype(int)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dname", required=True)
    p.add_argument("--tag", default="HSAT_attn", help="run tag used by train.py")
    p.add_argument("--donors", type=int, default=5)
    p.add_argument("--max_nodes", type=int, default=1000, help="test nodes per model (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", default=None, help="override the data_dir stored in the checkpoint")
    cli = p.parse_args()

    run_of = lambda s: int(s.rsplit("_run", 1)[1][:-3])
    paths = sorted(glob.glob(f"checkpoints/{cli.dname}_{cli.tag}_run*.pt"), key=run_of)
    assert paths, f"no checkpoints/{cli.dname}_{cli.tag}_run*.pt -- train with --save_ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targs = argparse.Namespace(**torch.load(paths[0], map_location="cpu", weights_only=False)["args"])
    if cli.data_dir:
        targs.data_dir = cli.data_dir
    assert targs.method == "HSAT" and (targs.mp_steps > 1 or targs.extract_subgraph), \
        "needs an HSAT model with a structure branch (mp_steps >= 2 or --extract_subgraph)"
    assert getattr(targs, "feat_source", "benchmark") != "label_trick", "label_trick models are not supported here"
    transform = torch_geometric.transforms.Compose([datasets.AddHypergraphSelfLoops()])
    data = datasets.HypergraphDataset(root=targs.data_dir, name=targs.dname, feature_noise='1',
                                      transform=transform, args=targs)._data.to(device)
    if getattr(targs, "feat_source", "benchmark") != "benchmark":
        data.x = torch.from_numpy(features.label_free_features(data, targs)).to(device)
    data.H = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1), device=device),
                                     size=(data.num_nodes, data.num_hyperedges)).coalesce().to(device)
    N = data.num_nodes
    log_deg = np.log1p(torch.bincount(data.edge_index[0], minlength=N).cpu().numpy())

    per_run, nodes_out = [], []
    pooled = {"dec": [], "drop_s": [], "drop_f": []}
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        run = run_of(path)
        model = GlobalHyperNodeTransformer(
            in_channels=data.x.shape[1], hidden_channels=targs.hidden_dim, out_channels=data.num_classes,
            n_layers=targs.n_layers, heads=targs.heads, dropout_rate=targs.dropout,
            edge_index=data.edge_index, hops=targs.hops, sparse_h=data.H, hg=None, args=targs).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        def probs(perturb=None):
            model.perturb = perturb
            with torch.no_grad():
                out = F.softmax(model(data), dim=1)
            model.perturb = None
            return out

        P = probs()
        yhat = P.argmax(1)
        with torch.no_grad():
            h0, h = model.saved_h0, model.saved_Xv
            g_fe = torch.sigmoid(model.gate_node_emb(h0))
            g_se = torch.sigmoid(model.gate_se(h))
            a, b = (g_fe * h0).norm(dim=1), (g_se * h).norm(dim=1)
            share = (b / (a + b).clamp_min(1e-12)).cpu().numpy()
            g_fe, g_se = g_fe.mean(1).cpu().numpy(), g_se.mean(1).cpu().numpy()

        rng = np.random.default_rng(cli.seed + run)
        test = ck["split"]["test"].numpy()
        if cli.max_nodes and len(test) > cli.max_nodes:
            test = rng.choice(test, cli.max_nodes, replace=False)
        drop_s, drop_sf, drop_f, flip_s, flip_f = (np.zeros(len(test)) for _ in range(5))
        for i, v in enumerate(test.tolist()):
            y, p0 = yhat[v].item(), P[v, yhat[v]].item()
            donors = rng.choice(N - 1, cli.donors, replace=False)
            donors = donors + (donors >= v)                      # never v itself
            for u in donors.tolist():
                ps = probs(dict(node=v, struct_from=u))[v]
                psf = probs(dict(node=v, struct_from=u, fixed_gate=True))[v]
                pf = probs(dict(node=v, feat_from=u))[v]
                drop_s[i] += (p0 - ps[y].item()) / cli.donors
                drop_sf[i] += (p0 - psf[y].item()) / cli.donors
                drop_f[i] += (p0 - pf[y].item()) / cli.donors
                flip_s[i] += (ps.argmax().item() != y) / cli.donors
                flip_f[i] += (pf.argmax().item() != y) / cli.donors

        gs, gf, sh, ld = g_se[test], g_fe[test], share[test], log_deg[test]
        diff = drop_s - drop_f
        mask = (diff != 0) & (sh != 0.5)
        r = {
            "mean drop, structure swap": float(drop_s.mean()),
            "mean drop, feature swap": float(drop_f.mean()),
            "flip rate, structure swap": float(flip_s.mean()),
            "flip rate, feature swap": float(flip_f.mean()),
            "rho(g_se, drop_s)": spearman(gs, drop_s),
            "rho(g_se, drop_s | log-degree)": partial_spearman(gs, drop_s, ld),
            "rho(share, drop_s)": spearman(sh, drop_s),
            "rho(share, drop_s | log-degree)": partial_spearman(sh, drop_s, ld),
            "rho(log-degree, drop_s)": spearman(ld, drop_s),
            "rho(g_se, drop_s), fixed gate": spearman(gs, drop_sf),
            "rho(g_fe, drop_f)": spearman(gf, drop_f),
            "rho(share, drop_s - drop_f)": spearman(sh, diff),
            "share predicts larger drop (%)": float(100 * np.mean(np.sign(sh[mask] - 0.5) == np.sign(diff[mask])))
                                              if mask.any() else float("nan"),
        }
        per_run.append(r)
        pooled["dec"] += deciles(gs).tolist()
        pooled["drop_s"] += drop_s.tolist()
        pooled["drop_f"] += drop_f.tolist()
        nodes_out.append(dict(run=run, nodes=test.tolist(), g_se=gs.tolist(), g_fe=gf.tolist(), share=sh.tolist(),
                              log_degree=ld.tolist(), drop_s=drop_s.tolist(), drop_s_fixed=drop_sf.tolist(),
                              drop_f=drop_f.tolist()))
        print(f"run {run}: {len(test)} nodes, rho(g_se, drop_s) = {r['rho(g_se, drop_s)']:.3f}", flush=True)

    print(f"\n== {cli.dname} / {cli.tag}: {len(per_run)} models, {cli.donors} donors per node ==")
    for k in per_run[0]:
        vals = np.array([r[k] for r in per_run], dtype=float)
        print(f"{k:40s} {np.nanmean(vals):+8.3f} ± {np.nanstd(vals):.3f}")
    dec, ds, dfe = (np.array(pooled[k]) for k in ("dec", "drop_s", "drop_f"))
    print("\nstructure-gate decile   mean drop, structure swap   mean drop, feature swap")
    for q in range(10):
        m = dec == q
        print(f"{q + 1:>12d} {ds[m].mean():>28.4f} {dfe[m].mean():>26.4f}")
    os.makedirs("results", exist_ok=True)
    out = f"results/gate_perturb_{cli.dname}_{cli.tag}.json"
    with open(out, "w") as f:
        json.dump({"cli": vars(cli), "per_run": per_run, "nodes": nodes_out}, f)
    print(f"\nper-node values written to {out}")


if __name__ == "__main__":
    main()
