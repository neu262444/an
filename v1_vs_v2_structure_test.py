#!/usr/bin/env python3
"""
V1 (efficient) vs V2 (expressive) on 1-GWL-indistinguishable hypergraphs with UNIFORM features.

Parts (choose with --parts, default: sep probe)
  sep    UNTRAINED. Share of node pairs whose encoder embeddings differ. Pairs are 1-GWL-equivalent
         by construction (base hypergraph vs. its 2-lift). Run with identity weights (deterministic,
         the large-scale version of Fig. 3 / Appendix B) and with S random weight draws.
  probe  Frozen random encoder + linear (logistic) probe. Nothing in the encoder is trained.
         Label = "is the node's K-hop sub-hypergraph Berge-acyclic?" (depends on structure only).
  e2e    (optional) Full pipeline trained end-to-end: encoder -> gate (Eq. 9) -> attention -> head.
         Slower, and can be unstable without a Phi residual; use for the appendix only if stable.

Reference classifiers (no neural net) are printed next to the probe so the result can be read honestly:
  majority class | 1-GWL colour classifier (what ANY message-passing HGNN can reach) | |V_v| classifier.

Notes
  * Self-contained implementation of Eqs. (9)-(12) and Set-Union Pooling used for Appendix B (Tables 6-8); it does not import models/HSAT.py.
  * With uniform features and mean aggregation, V1 gives the SAME vector for every node of every
    hypergraph, so V1 sits exactly at the majority rate by construction.
  * V2 set-union with mean aggregation reduces to |V_v| * c under uniform features, i.e. it counts
    the K-hop ball. The label here is a function of that count. Claim "beyond 1-GWL", not "beyond counting".
  * V2 root readout needs sum-type aggregation and L > K to see anything under uniform features.
"""
import argparse
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


# ------------------------------------------------------------------ data generation
def biregular(n, d, r, rng):
    m = n * d // r
    assert m * r == n * d, "n*d must be divisible by r"
    for _ in range(2000):
        stubs = [v for v in range(n) for _ in range(d)]
        rng.shuffle(stubs)
        edges = [stubs[i * r:(i + 1) * r] for i in range(m)]
        if all(len(set(e)) == r for e in edges) and len({tuple(sorted(e)) for e in edges}) == m:
            return edges
    raise RuntimeError("could not sample a simple biregular hypergraph")


def two_lift(n, edges, rng):
    """Covering map => every lifted node has the same unrolled computation tree as its base node."""
    out = []
    for e in edges:
        s = [rng.randint(0, 1) for _ in e]
        out.append([v + n * t for v, t in zip(e, s)])
        out.append([v + n * (1 - t) for v, t in zip(e, s)])
    return out


def khop(v, K, n2e, edges):
    nodes, frontier, used, seen = {v}, [v], [], set()
    for _ in range(K):
        nxt = []
        for u in frontier:
            for e in n2e[u]:
                if e in seen:
                    continue
                seen.add(e)
                used.append(e)
                for w in edges[e]:
                    if w not in nodes:
                        nodes.add(w)
                        nxt.append(w)
        frontier = nxt
    return sorted(nodes), used


class Instance:
    def __init__(self, N, edges, pairs, K):
        self.N, self.edges, self.M = N, edges, len(edges)
        self.n2e = [[] for _ in range(N)]
        for j, e in enumerate(edges):
            for v in e:
                self.n2e[v].append(j)
        self.vi = torch.tensor([v for e in edges for v in e])
        self.ei = torch.tensor([j for j, e in enumerate(edges) for _ in e])
        svi, sei, node_map, root_id, ball, acyc = [], [], [], [], [], []
        nb = ne = 0
        for v in range(N):
            nodes, used = khop(v, K, self.n2e, edges)
            loc = {u: i for i, u in enumerate(nodes)}
            for j, e in enumerate(used):
                for u in edges[e]:
                    svi.append(nb + loc[u])
                    sei.append(ne + j)
            node_map += nodes
            root_id += [v] * len(nodes)
            ball.append(len(nodes))
            n_inc = sum(len(edges[e]) for e in used)
            acyc.append(int(n_inc == len(nodes) + len(used) - 1))  # tree <=> incidences = V + E - 1
            nb += len(nodes)
            ne += len(used)
        self.svi, self.sei = torch.tensor(svi), torch.tensor(sei)
        self.node_map, self.root_id = torch.tensor(node_map), torch.tensor(root_id)
        self.n_sub, self.m_sub = nb, ne
        self.ball, self.y = torch.tensor(ball), torch.tensor(acyc)
        self.pairs = torch.tensor(pairs)
        self.pair_diff = torch.tensor([ball[a] != ball[b] for a, b in pairs])


def make_cover_instance(n, d, r, K, rng):
    base = biregular(n, d, r, rng)
    lift = two_lift(n, base, rng)
    edges = base + [[v + n for v in e] for e in lift]
    pairs = [(i, n + i + n * s) for i in range(n) for s in (0, 1)]
    return Instance(3 * n, edges, pairs, K)


def appendix_b_instance(K):
    hg1 = [[0, 1, 2], [1, 3, 4], [4, 5, 6], [2, 7, 6]]          # v,b,c,d,e,f,g,h = 0..7
    hg2 = [[8, 9, 10], [9, 10, 11], [12, 13, 14], [13, 14, 15]]  # u,b,c,d,e,f,g,h = 8..15
    return Instance(16, hg1 + hg2, [(0, 8)], K)


def build_dataset(n_inst, sizes, d, r, K, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n_inst:
        inst = make_cover_instance(rng.choice(sizes), d, r, K, rng)
        if inst.pair_diff.any():
            out.append(inst)
    return out


def wl_colors(insts, rounds=8):
    table = {}

    def intern(key):
        return table.setdefault(key, len(table))

    state = [([0] * I.N, [1] * I.M) for I in insts]
    for _ in range(rounds):
        new = []
        for I, (nc, ec) in zip(insts, state):
            nc2 = [intern((nc[v], tuple(sorted(ec[e] for e in I.n2e[v])))) for v in range(I.N)]
            ec2 = [intern((ec[e], tuple(sorted(nc[v] for v in I.edges[e])))) for e in range(I.M)]
            new.append((nc2, ec2))
        state = new
    return [nc for nc, _ in state]


class Batch:
    pass


def collate(insts):
    b, N, M, NS, MS = Batch(), 0, 0, 0, 0
    cols = defaultdict(list)
    for gi, I in enumerate(insts):
        cols["vi"].append(I.vi + N); cols["ei"].append(I.ei + M)
        cols["svi"].append(I.svi + NS); cols["sei"].append(I.sei + MS)
        cols["node_map"].append(I.node_map + N); cols["root_id"].append(I.root_id + N)
        cols["gid"].append(torch.full((I.N,), gi)); cols["y"].append(I.y); cols["ball"].append(I.ball)
        cols["pairs"].append(I.pairs + N); cols["pair_diff"].append(I.pair_diff)
        N, M, NS, MS = N + I.N, M + I.M, NS + I.n_sub, MS + I.m_sub
    for k, v in cols.items():
        setattr(b, k, torch.cat(v))
    b.N, b.M, b.n_sub, b.m_sub = N, M, NS, MS
    b.mask = b.gid[:, None] == b.gid[None, :]
    return b


def batches_of(insts, bs):
    return [collate(insts[i:i + bs]) for i in range(0, len(insts), bs)]


# ------------------------------------------------------------------ encoders
class HGNN(nn.Module):
    """Scatter-gather, Eqs. (10)-(11). agg='mean' = paper's aggregator (W2, W4 = identity).
    agg='sum' = size/degree-aware 1-GWL reference. identity=True -> W1 = W3 = identity (no parameters)."""

    def __init__(self, d, agg, identity=False):
        super().__init__()
        self.agg = agg
        if identity:
            self.W1, self.W3 = nn.Identity(), nn.Identity()
        else:
            self.W1 = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
            self.W3 = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def layer(self, h, vi, ei, nv, ne):
        d = h.size(1)
        msg = torch.zeros(ne, d).index_add(0, ei, self.W1(h)[vi])
        if self.agg == "mean":
            msg = msg / torch.bincount(ei, minlength=ne).clamp(min=1).unsqueeze(1)
        he = self.W3(msg)
        out = torch.zeros(nv, d).index_add(0, vi, he[ei])
        if self.agg == "mean":
            out = out / torch.bincount(vi, minlength=nv).clamp(min=1).unsqueeze(1)
        return out


class Encoder(nn.Module):
    def __init__(self, d, kind, agg, readout, K, L, identity=False):
        super().__init__()
        self.kind, self.readout, self.K, self.L = kind, readout, K, L
        self.hgnn = HGNN(d, agg, identity)

    def forward(self, h0, b):
        if self.kind == "v1":
            h, acc = h0, 0
            for _ in range(self.K):
                h = self.hgnn.layer(h, b.vi, b.ei, b.N, b.M)
                acc = acc + h
            return acc / self.K
        hs = h0[b.node_map]
        for _ in range(self.L):
            hs = self.hgnn.layer(hs, b.svi, b.sei, b.n_sub, b.m_sub)
        if self.readout == "set":
            return torch.zeros(b.N, hs.size(1)).index_add(0, b.root_id, hs)
        is_root = b.node_map == b.root_id
        return torch.zeros(b.N, hs.size(1)).index_copy(0, b.root_id[is_root], hs[is_root])


ENCODERS = {  # name: (kind, agg, readout)
    "V1 (efficient, mean)": ("v1", "mean", None),
    "V1 (sum, 1-GWL ref.)": ("v1", "sum", None),
    "V2 (root readout)": ("v2", "mean", "root"),
    "V2 (set-union readout)": ("v2", "mean", "set"),
}
D_IN = 8


def initial_features(b, d, win):
    """Uniform raw features. win=None -> identity weights, h0 = 1; else h0 = tanh(W_in x), x = 1."""
    if win is None:
        return torch.ones(b.N, d)
    return torch.tanh(win(torch.ones(b.N, D_IN)))


def make_encoder(spec, d, K, L, seed):
    """seed=None -> identity weights (deterministic); otherwise random weights from this seed."""
    kind, agg, readout = spec
    if seed is None:
        return Encoder(d, kind, agg, readout, K, L, identity=True), None
    torch.manual_seed(seed)
    win = nn.Linear(D_IN, d)
    return Encoder(d, kind, agg, readout, K, L), win


# ------------------------------------------------------------------ Part 0: Appendix-B check
def appendix_b_check(K, L):
    ab = appendix_b_instance(K)
    b = collate([ab])
    wl = wl_colors([ab], 8)[0]
    print("== Part 0: Appendix-B pair, identity weights, unit initialisation (d=1) ==")
    print(f"   1-WL colours equal: {wl[0] == wl[8]}   |V_v|={int(ab.ball[0])}   |V_u|={int(ab.ball[8])}")
    for name, spec in ENCODERS.items():
        enc, _ = make_encoder(spec, 1, K, L, None)
        with torch.no_grad():
            h = enc(torch.ones(b.N, 1), b)[:, 0]
        print(f"   {name:24s} h_v = {h[0].item():8.3f}   h_u = {h[8].item():8.3f}")
    print()


# ------------------------------------------------------------------ Part A: untrained separation
def separation_rates(enc, win, batches, d):
    seps, diffs = [], []
    with torch.no_grad():
        for b in batches:
            h = enc(initial_features(b, d, win), b)
            seps.append((h[b.pairs[:, 0]] - h[b.pairs[:, 1]]).norm(dim=1) > 1e-9)
            diffs.append(b.pair_diff)
    s, dm = torch.cat(seps), torch.cat(diffs)
    return 100 * s[dm].double().mean().item(), 100 * s[~dm].double().mean().item() if (~dm).any() else float("nan")


def part_sep(batches, args):
    print("== Part A (UNTRAINED): % of 1-WL-equivalent pairs with different embeddings ==")
    print(f"{'encoder':24s} {'weights':18s} {'|V_v| differs':>14s} {'|V_v| equal':>12s}")
    for name, spec in ENCODERS.items():
        enc, win = make_encoder(spec, args.d, args.K, args.L, None)
        a, c = separation_rates(enc, win, batches, args.d)
        print(f"{name:24s} {'identity':18s} {a:13.1f}% {c:11.1f}%")
        res = []
        for seed in range(args.seeds):
            enc, win = make_encoder(spec, args.d, args.K, args.L, seed)
            res.append(separation_rates(enc, win, batches, args.d))
        r = np.array(res)
        print(f"{'':24s} {f'random x{args.seeds}':18s} {r[:, 0].mean():8.1f} +- {r[:, 0].std():3.1f}% {np.nanmean(r[:, 1]):10.1f}%")
    print("   ('|V_v| differs' = pairs whose K-hop balls have different size; 'equal' = control, expected 0.)\n")


# ------------------------------------------------------------------ Part B: frozen encoder + linear probe
def plug_in_accuracy(train_feat, train_y, test_feat, test_y):
    cnt = defaultdict(lambda: np.zeros(2))
    for f, y in zip(train_feat, train_y):
        cnt[f][y] += 1
    glob = np.bincount(train_y, minlength=2).argmax()
    pred = [cnt[f].argmax() if f in cnt else glob for f in test_feat]
    return float(np.mean(np.array(pred) == np.array(test_y)))


def reference_accuracies(train, test):
    ytr = torch.cat([I.y for I in train]).numpy()
    yte = torch.cat([I.y for I in test]).numpy()
    cols = wl_colors(train + test, 8)
    wl_flat = [c for col in cols for c in col]
    ball_flat = [int(x) for I in train + test for x in I.ball]
    ntr = sum(I.N for I in train)
    return {
        "majority class": float(max(yte.mean(), 1 - yte.mean())),
        "1-WL colour classifier (any MPNN)": plug_in_accuracy(wl_flat[:ntr], ytr, wl_flat[ntr:], yte),
        "|V_v| classifier (ball size)": plug_in_accuracy(ball_flat[:ntr], ytr, ball_flat[ntr:], yte),
    }


def logistic_probe(Xtr, ytr, Xte, yte, l2=1e-4):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 1e-9 * (1 + mu.abs())            # drop (numerically) constant columns
    if keep.sum() == 0:                          # constant features -> only the bias is usable -> majority
        return (yte == torch.bincount(ytr, minlength=2).argmax()).double().mean().item()
    Xtr, Xte = (Xtr[:, keep] - mu[keep]) / sd[keep], (Xte[:, keep] - mu[keep]) / sd[keep]
    lin = nn.Linear(Xtr.size(1), 2)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(Xtr), ytr) + l2 * sum((p ** 2).sum() for p in lin.parameters())
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (lin(Xte).argmax(1) == yte).double().mean().item()


def part_probe(train, test, args):
    tr_b, te_b = batches_of(train, args.bs), batches_of(test, args.bs)
    ytr, yte = torch.cat([b.y for b in tr_b]), torch.cat([b.y for b in te_b])
    refs = reference_accuracies(train, test)
    print("== Part B: frozen (untrained) encoder + logistic-regression probe ==")
    print("reference classifiers on the test set:")
    for k, v in refs.items():
        print(f"   {k:36s} {100 * v:6.2f}")
    print(f"\n{'encoder (frozen, random weights)':36s} probe test acc. % (mean +- std over {args.seeds} encoder draws)")
    for name, spec in ENCODERS.items():
        accs = []
        for seed in range(args.seeds):
            enc, win = make_encoder(spec, args.d, args.K, args.L, seed)
            with torch.no_grad():
                Xtr = torch.cat([enc(initial_features(b, args.d, win), b) for b in tr_b])
                Xte = torch.cat([enc(initial_features(b, args.d, win), b) for b in te_b])
            accs.append(logistic_probe(Xtr, ytr, Xte, yte))
        print(f"{name:36s} {100 * np.mean(accs):6.2f} +- {100 * np.std(accs):.2f}")
    print()


# ------------------------------------------------------------------ Part C (optional): end-to-end
class GatedAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.psi1, self.psi2 = nn.Linear(d, d), nn.Linear(d, d)
        self.Wq, self.Wk, self.Wv = (nn.Linear(d, d, bias=False) for _ in range(3))

    def forward(self, x0, h, mask):
        phi = torch.sigmoid(self.psi1(x0)) * x0 + torch.sigmoid(self.psi2(h)) * h   # Eq. (9)
        s = (self.Wq(phi) @ self.Wk(phi).T / math.sqrt(self.d)).masked_fill(~mask, float("-inf"))
        return phi, s.softmax(-1) @ self.Wv(phi)                                      # Q, K, V all from Phi


class Model(nn.Module):
    def __init__(self, spec, d, K, L, residual):
        super().__init__()
        kind, agg, readout = spec
        self.d, self.residual = d, residual
        self.Win = nn.Linear(D_IN, d)
        self.enc = Encoder(d, kind, agg, readout, K, L)
        self.att = GatedAttention(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, b):
        h0 = torch.tanh(self.Win(torch.ones(b.N, D_IN)))
        phi, o = self.att(h0, self.enc(h0, b), b.mask)
        return self.head(o + phi if self.residual else o)


def part_e2e(train, val, test, args):
    tr, va, te = batches_of(train, args.bs), batches_of(val, args.bs), batches_of(test, args.bs)

    def acc(model, batches):
        ok = n = 0
        with torch.no_grad():
            for b in batches:
                ok += (model(b).argmax(1) == b.y).sum().item()
                n += b.N
        return ok / n

    print(f"== Part C (optional): end-to-end training, residual={'Phi' if args.residual else 'none'} ==")
    for name, spec in ENCODERS.items():
        if name == "V1 (sum, 1-GWL ref.)":
            continue
        accs, t0 = [], time.time()
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = Model(spec, args.d, args.K, args.L, args.residual)
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
        print(f"{name:24s} {100 * np.mean(accs):6.2f} +- {100 * np.std(accs):.2f}   "
              f"(min {100 * np.min(accs):.2f}, max {100 * np.max(accs):.2f}; {time.time() - t0:.0f}s)")
    print()


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", default=["sep", "probe"], choices=["sep", "probe", "e2e"])
    p.add_argument("--n-inst", type=int, default=600)
    p.add_argument("--sizes", type=int, nargs="+", default=[12, 15, 18, 21, 24, 27, 30],
                   help="base-hypergraph node counts (multiples of 3 for (2,3)-biregular)")
    p.add_argument("--deg", type=int, default=2)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--K", type=int, default=2, help="BFS hops (V2) / hops (V1). Larger K -> more pairs separated.")
    p.add_argument("--L", type=int, default=3, help="HGNN layers inside each sub-hypergraph (root readout needs L > K)")
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--residual", action="store_true", help="e2e only: add Phi residual after attention")
    p.add_argument("--seed", type=int, default=0, help="dataset seed")
    args = p.parse_args()

    appendix_b_check(args.K, args.L)

    data = build_dataset(args.n_inst, args.sizes, args.deg, args.rank, args.K, args.seed)
    cols = wl_colors(data, 8)
    ok = all(cols[i][a.item()] == cols[i][c.item()] for i, I in enumerate(data) for a, c in I.pairs)
    n_pairs = sum(len(I.pairs) for I in data)
    n_diff = sum(int(I.pair_diff.sum()) for I in data)
    print(f"== Dataset: {len(data)} instances, base n in {args.sizes}, ({args.deg},{args.rank})-biregular, K={args.K} ==")
    print(f"   all pairs 1-WL-equivalent: {ok}   pairs: {n_pairs}   |V_v| differs: {n_diff} ({100 * n_diff / n_pairs:.0f}%)")
    print(f"   share of nodes with acyclic K-hop ball (label=1): {torch.cat([I.y for I in data]).double().mean().item():.2f}\n")
    assert ok, "pairs are not 1-WL-equivalent -- generator bug"

    n = len(data)
    a, c = int(0.6 * n), int(0.8 * n)
    train, val, test = data[:a], data[a:c], data[c:]

    if "sep" in args.parts:
        part_sep(batches_of(data, args.bs), args)
    if "probe" in args.parts:
        part_probe(train, test, args)
    if "e2e" in args.parts:
        part_e2e(train, val, test, args)


if __name__ == "__main__":
    main()