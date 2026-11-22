#!/usr/bin/env python3
"""
Q/K vs Q/K/V ablation on the large synthetic dataset (uniform features, 1-WL-equivalent pairs).

Needs v1_vs_v2_structure_test.py in the same directory (data generator, encoders, references).

One gated-attention block, two switches:
    V source : "raw" -> V = W_V h0        (h0 = tanh(W_in x), structure-blind)
               "phi" -> V = W_V Phi       (Phi = gate(h0, h), Eq. 9, structure-aware)
    skip     : "none" | "raw" (adds h0, structure-blind) | "phi" (adds Phi, structure-carrying)
Q and K always come from Phi. The encoder is the same in all variants.

    id   V     skip   meaning
    a    raw   raw    SAT-style: structure only in the attention weights, structure-blind skip
    b    raw   phi    structure reaches the output through the skip but not through V
    c    phi   phi    full model (Q, K, V and skip from Phi)
    d    phi   raw    structure reaches the output only through V
    a0   raw   none   attention operator only, structure only in Q/K
    c0   phi   none   attention operator only, structure in Q/K/V

Contrasts
    a  vs d  : effect of structural V when the skip is structure-blind        (Theorem 1, inside a block)
    b  vs c  : effect of structural V when the skip already carries structure (the real model's setting)
    a  vs b  : effect of a structure-carrying skip
    a0 vs c0 : effect of structural V on the bare attention operator

Parts (--parts)
    dist   UNTRAINED: share of 1-WL-equivalent pairs (with different |V_v|) whose block outputs differ,
           and the median normalised distance, over random initialisations.
    train  end-to-end training, test accuracy over seeds (mean +- std, min-max).
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import v1_vs_v2_structure_test as base

torch.set_default_dtype(torch.float64)

VARIANTS = {
    "a": ("(a)  QK-only, raw V,  raw skip   [SAT-style]", "raw", "raw"),
    "b": ("(b)  QK-only, raw V,  Phi skip", "raw", "phi"),
    "c": ("(c)  full QKV,        Phi skip   [full model]", "phi", "phi"),
    "d": ("(d)  QKV,             raw skip   [structure only via V]", "phi", "raw"),
    "a0": ("(a0) QK-only, raw V,  no skip    [operator only]", "raw", "none"),
    "c0": ("(c0) full QKV,        no skip    [operator only]", "phi", "none"),
}
ENC_KEYS = {
    "v1": "V1 (efficient, mean)",
    "v2root": "V2 (root readout)",
    "v2set": "V2 (set-union readout)",
}


class Block(nn.Module):
    """Eq. (9) gate + single-head attention with selectable V source and skip source.
    Parameter creation order matches base.GatedAttention, so variant (c) reproduces the earlier run."""

    def __init__(self, d, v_src, skip):
        super().__init__()
        self.d, self.v_src, self.skip = d, v_src, skip
        self.psi1, self.psi2 = nn.Linear(d, d), nn.Linear(d, d)
        self.Wq, self.Wk, self.Wv = (nn.Linear(d, d, bias=False) for _ in range(3))

    def forward(self, h0, h, mask):
        phi = torch.sigmoid(self.psi1(h0)) * h0 + torch.sigmoid(self.psi2(h)) * h
        s = (self.Wq(phi) @ self.Wk(phi).T / math.sqrt(self.d)).masked_fill(~mask, float("-inf"))
        o = s.softmax(-1) @ self.Wv(phi if self.v_src == "phi" else h0)
        if self.skip == "phi":
            o = o + phi
        elif self.skip == "raw":
            o = o + h0
        return o


class Model2(nn.Module):
    def __init__(self, spec, d, K, L, v_src, skip):
        super().__init__()
        kind, agg, readout = spec
        self.Win = nn.Linear(base.D_IN, d)
        self.enc = base.Encoder(d, kind, agg, readout, K, L)
        self.att = Block(d, v_src, skip)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))

    def repr(self, b):
        h0 = torch.tanh(self.Win(torch.ones(b.N, base.D_IN)))   # uniform features -> uniform h0
        return self.att(h0, self.enc(h0, b), b.mask)

    def forward(self, b):
        return self.head(self.repr(b))


# ------------------------------------------------------------------ Part D1: untrained
def norm_pair_dist(X, b):
    """raw pair distance and distance divided by the mean pairwise distance inside the instance."""
    a, c = b.pairs[:, 0], b.pairs[:, 1]
    raw = (X[a] - X[c]).norm(dim=1)
    dens, off = [], 0
    for n in torch.bincount(b.gid).tolist():
        Xi = X[off:off + n]
        dens.append(torch.cdist(Xi, Xi).sum() / (n * (n - 1)))
        off += n
    den = torch.stack(dens)[b.gid[a]]
    norm = torch.where(den > 1e-9, raw / den.clamp(min=1e-9), torch.zeros_like(raw))
    return raw, norm


def part_dist(data, args):
    batches = base.batches_of(data[:args.dist_inst], args.bs)
    print(f"== Part D1 (UNTRAINED): block outputs of 1-WL-equivalent pairs with different |V_v| "
          f"({args.dist_inst} instances, {args.seeds} random inits) ==")
    print(f"{'encoder':24s} {'variant':54s} {'pairs separated':>16s} {'median norm. dist.':>20s}")
    for ek in args.encoders:
        for vid in args.variants:
            label, v_src, skip = VARIANTS[vid]
            seps, nrms = [], []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                model = Model2(base.ENCODERS[ENC_KEYS[ek]], args.d, args.K, args.L, v_src, skip).eval()
                raws, ns, dms = [], [], []
                with torch.no_grad():
                    for b in batches:
                        raw, nrm = norm_pair_dist(model.repr(b), b)
                        raws.append(raw); ns.append(nrm); dms.append(b.pair_diff)
                raw, nrm, dm = torch.cat(raws), torch.cat(ns), torch.cat(dms)
                seps.append(100 * (raw[dm] > 1e-9).double().mean().item())
                nrms.append(nrm[dm].median().item())
            print(f"{ENC_KEYS[ek]:24s} {label:54s} {np.mean(seps):9.1f} +- {np.std(seps):3.1f}% "
                  f"{np.mean(nrms):12.3f} +- {np.std(nrms):.3f}")
    print("   (separated = distance > 1e-9. Exactly 0.0% means the output is identical for both nodes.)\n")


# ------------------------------------------------------------------ Part D2: training
def part_train(train, val, test, args):
    tr, va, te = (base.batches_of(x, args.bs) for x in (train, val, test))
    refs = base.reference_accuracies(train, test)
    print("== Part D2: end-to-end training (uniform features, structure-only label) ==")
    print("reference classifiers on the test set:")
    for k, v in refs.items():
        print(f"   {k:36s} {100 * v:6.2f}")
    print()

    def acc(model, batches):
        ok = n = 0
        with torch.no_grad():
            for b in batches:
                ok += (model(b).argmax(1) == b.y).sum().item()
                n += b.N
        return ok / n

    for ek in args.encoders:
        print(f"encoder: {ENC_KEYS[ek]}  | {args.seeds} seeds, {args.epochs} epochs, lr {args.lr}")
        print(f"{'variant':54s} {'test acc. % (mean +- std)':>26s}  {'min':>6s} {'max':>6s}")
        for vid in args.variants:
            label, v_src, skip = VARIANTS[vid]
            accs, t0 = [], time.time()
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                model = Model2(base.ENCODERS[ENC_KEYS[ek]], args.d, args.K, args.L, v_src, skip)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                best_va, best_te = -1.0, 0.0
                for ep in range(args.epochs):
                    model.train()
                    for i in np.random.RandomState(seed * 1000 + ep).permutation(len(tr)):
                        loss = F.cross_entropy(model(tr[i]), tr[i].y)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    model.eval()
                    v = acc(model, va)
                    if v > best_va:
                        best_va, best_te = v, acc(model, te)
                accs.append(best_te)
            print(f"{label:54s} {100 * np.mean(accs):17.2f} +- {100 * np.std(accs):5.2f}  "
                  f"{100 * np.min(accs):6.2f} {100 * np.max(accs):6.2f}   ({time.time() - t0:.0f}s)", flush=True)
        print()


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", default=["dist", "train"], choices=["dist", "train"])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--encoders", nargs="+", default=["v2set"], choices=list(ENC_KEYS),
                   help="v2set is the informative one; v1 / v2root should stay at majority in every variant")
    p.add_argument("--n-inst", type=int, default=600)
    p.add_argument("--dist-inst", type=int, default=200)
    p.add_argument("--sizes", type=int, nargs="+", default=[12, 15, 18, 21, 24, 27, 30])
    p.add_argument("--deg", type=int, default=2)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0, help="dataset seed")
    args = p.parse_args()

    data = base.build_dataset(args.n_inst, args.sizes, args.deg, args.rank, args.K, args.seed)
    cols = base.wl_colors(data, 8)
    assert all(cols[i][a.item()] == cols[i][c.item()] for i, I in enumerate(data) for a, c in I.pairs), \
        "pairs are not 1-WL-equivalent -- generator bug"
    n_pairs = sum(len(I.pairs) for I in data)
    n_diff = sum(int(I.pair_diff.sum()) for I in data)
    print(f"== Dataset: {len(data)} instances, K={args.K}; pairs 1-WL-equivalent: True; "
          f"|V_v| differs for {n_diff}/{n_pairs} pairs ({100 * n_diff / n_pairs:.0f}%) ==\n")

    n = len(data)
    a, c = int(0.6 * n), int(0.8 * n)
    if "dist" in args.parts:
        part_dist(data, args)
    if "train" in args.parts:
        part_train(data[:a], data[a:c], data[c:], args)


if __name__ == "__main__":
    main()