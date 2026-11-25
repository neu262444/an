#!/usr/bin/env python3
"""
Structure-only synthetic test for HSGF-T
=========================================
Uniform node features + node pairs that (a) are 1-GWL-equivalent (so no message-passing HGNN
can separate them) but (b) have different K-hop sub-hypergraphs.

What it measures
  [Part 0] Appendix-B pair (HG1 vs HG2) reproduced as a sanity check.
  [Part 1] Distances between the two nodes of each pair, at four points of the pipeline:
             h    = structural encoder output          (||h_v - h_u||)
             phi  = after the gate, Eq. (9)            (||Phi_v - Phi_u||)
             attn = attention output, BEFORE residual  (||o_v - o_u||)       <- Theorem 1 / 3
             attn+res = attention output + Phi residual
           for 5 encoders x 2 attention variants (structure in Q,K,V  vs  Q,K only).
           Raw and normalised (divided by mean pairwise distance inside the instance).
  [Part 2] Node classification where the label depends only on structure
           (label = "is the K-hop sub-hypergraph Berge-acyclic?"), features uniform.


Usage
  python synthetic_structure_test.py --mode dist                 # fast, no training
  python synthetic_structure_test.py --mode train --epochs 40    # slower
  python synthetic_structure_test.py --mode all
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)  # float64 so exact zeros are visibly ~1e-16, not 1e-8


# ----------------------------------------------------------------------------------------
# 1. Hypergraph generation:  base hypergraph B  +  its random 2-lift L  (disjoint union)
# ----------------------------------------------------------------------------------------
def biregular(n, d, r, rng):
    """Random (d, r)-biregular hypergraph: every node in d hyperedges, every hyperedge of size r."""
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
    """Random 2-lift: node (i,s) -> i + n*s. Each hyperedge e becomes two edges that use
    opposite 'sheets' for each member. This is a covering map, so every lifted node has the
    same unrolled computation tree as its base node (=> 1-GWL cannot tell them apart), but
    short cycles in the base get unwrapped, so K-hop sub-hypergraphs differ."""
    out = []
    for e in edges:
        s = [rng.randint(0, 1) for _ in e]
        out.append([v + n * t for v, t in zip(e, s)])
        out.append([v + n * (1 - t) for v, t in zip(e, s)])
    return out


def khop(v, K, n2e, edges):
    """K rounds of BFS; every traversed hyperedge is kept whole (as in Sec. 4.1, Variant 2)."""
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
    """One hypergraph (disjoint union) + node pairs + precomputed K-hop sub-hypergraphs."""

    def __init__(self, N, edges, pairs, K):
        self.N, self.edges, self.M, self.K = N, edges, len(edges), K
        self.n2e = [[] for _ in range(N)]
        for j, e in enumerate(edges):
            for v in e:
                self.n2e[v].append(j)
        # full incidence (scatter/gather index lists)
        self.vi = torch.tensor([v for e in edges for v in e])
        self.ei = torch.tensor([j for j, e in enumerate(edges) for _ in e])
        # per-root sub-hypergraphs, stored as ONE big block-diagonal incidence
        svi, sei, node_map, root_id = [], [], [], []
        nb = ne = 0
        ball, acyc = [], []
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
            acyc.append(int(n_inc == len(nodes) + len(used) - 1))  # connected + (inc = V+E-1) <=> tree
            nb += len(nodes)
            ne += len(used)
        self.svi, self.sei = torch.tensor(svi), torch.tensor(sei)
        self.node_map, self.root_id = torch.tensor(node_map), torch.tensor(root_id)
        self.n_sub, self.m_sub = nb, ne
        self.ball = torch.tensor(ball)
        self.y = torch.tensor(acyc)
        self.pairs = torch.tensor(pairs)
        self.pair_diff = torch.tensor([ball[a] != ball[b] for a, b in pairs])


def make_cover_instance(n, d, r, K, rng):
    base = biregular(n, d, r, rng)
    lift = two_lift(n, base, rng)
    edges = base + [[v + n for v in e] for e in lift]  # lift lives on nodes n .. 3n-1
    pairs = [(i, n + i + n * s) for i in range(n) for s in (0, 1)]  # (base node, its two preimages)
    return Instance(3 * n, edges, pairs, K)


def appendix_b_instance(K=2):
    """HG1 (nodes 0-7: v,b,c,d,e,f,g,h) and HG2 (nodes 8-15: u,b,c,d,e,f,g,h) from Appendix B."""
    hg1 = [[0, 1, 2], [1, 3, 4], [4, 5, 6], [2, 7, 6]]
    hg2 = [[8, 9, 10], [9, 10, 11], [12, 13, 14], [13, 14, 15]]
    return Instance(16, hg1 + hg2, [(0, 8)], K)


def build_dataset(n_inst, sizes, d, r, K, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n_inst:
        inst = make_cover_instance(rng.choice(sizes), d, r, K, rng)
        if inst.pair_diff.any():  # keep instances with at least one structurally different pair
            out.append(inst)
    return out


# ----------------------------------------------------------------------------------------
# 1b. 1-WL colour refinement on the incidence graph (verifies pairs are message-passing-equivalent)
# ----------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------
# 2. Batching (block-diagonal; attention restricted to the same instance)
# ----------------------------------------------------------------------------------------
class Batch:
    pass


def collate(insts):
    b, N, M, NS, MS = Batch(), 0, 0, 0, 0
    cols = defaultdict(list)
    b.slices = []
    for gi, I in enumerate(insts):
        cols["vi"].append(I.vi + N); cols["ei"].append(I.ei + M)
        cols["svi"].append(I.svi + NS); cols["sei"].append(I.sei + MS)
        cols["node_map"].append(I.node_map + N); cols["root_id"].append(I.root_id + N)
        cols["gid"].append(torch.full((I.N,), gi)); cols["y"].append(I.y); cols["ball"].append(I.ball)
        cols["pairs"].append(I.pairs + N); cols["pair_diff"].append(I.pair_diff)
        cols["pair_gid"].append(torch.full((len(I.pairs),), gi))
        b.slices.append((N, I.N))
        N, M, NS, MS = N + I.N, M + I.M, NS + I.n_sub, MS + I.m_sub
    for k, v in cols.items():
        setattr(b, k, torch.cat(v))
    b.N, b.M, b.n_sub, b.m_sub = N, M, NS, MS
    b.mask = b.gid[:, None] == b.gid[None, :]
    return b


# ----------------------------------------------------------------------------------------
# 3. Model: structural encoder -> gate (Eq. 9) -> single-head attention
# ----------------------------------------------------------------------------------------
class HGNN(nn.Module):
    """Scatter-gather layer, Eqs. (10)-(11). agg='mean' is the paper's aggregator (W2, W4 = id);
    agg='sum' is a degree/size-aware 1-GWL reference (what an ED-HNN/AllDeepSets-style MPNN can do)."""

    def __init__(self, d, agg):
        super().__init__()
        self.agg = agg
        self.W1 = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.W3 = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def layer(self, h, vi, ei, nv, ne):
        d = h.size(1)
        msg = torch.zeros(ne, d).index_add(0, ei, self.W1(h)[vi])           # node -> hyperedge
        if self.agg == "mean":
            msg = msg / torch.bincount(ei, minlength=ne).clamp(min=1).unsqueeze(1)
        he = self.W3(msg)
        out = torch.zeros(nv, d).index_add(0, vi, he[ei])                   # hyperedge -> node
        if self.agg == "mean":
            out = out / torch.bincount(vi, minlength=nv).clamp(min=1).unsqueeze(1)
        return out


class Encoder(nn.Module):
    """kind='v1': K-hop multi-hop aggregator, hop-averaged (Eq. 12).
       kind='v2': per-node K-hop sub-hypergraph, L HGNN layers inside it, readout 'root' | 'set'."""

    def __init__(self, d, kind, agg, readout, K, L):
        super().__init__()
        self.kind, self.readout, self.K, self.L = kind, readout, K, L
        self.hgnn = HGNN(d, agg)

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
        if self.readout == "set":                                           # set-union pooling (sum)
            return torch.zeros(b.N, hs.size(1)).index_add(0, b.root_id, hs)
        is_root = b.node_map == b.root_id                                   # root readout
        return torch.zeros(b.N, hs.size(1)).index_copy(0, b.root_id[is_root], hs[is_root])


class GatedAttention(nn.Module):
    """Eq. (9) gate + single-head softmax attention.
       vmode='qkv': Q,K,V all from Phi (HSA-attn).  vmode='qk': Q,K from Phi, V from raw features
       (structure only in the attention weights, the SAT-style design)."""

    def __init__(self, d, vmode):
        super().__init__()
        self.vmode, self.d = vmode, d
        self.psi1, self.psi2 = nn.Linear(d, d), nn.Linear(d, d)
        self.Wq, self.Wk, self.Wv = (nn.Linear(d, d, bias=False) for _ in range(3))

    def forward(self, x0, h, mask):
        phi = torch.sigmoid(self.psi1(x0)) * x0 + torch.sigmoid(self.psi2(h)) * h
        s = self.Wq(phi) @ self.Wk(phi).T / math.sqrt(self.d)
        s = s.masked_fill(~mask, float("-inf"))
        v = self.Wv(phi if self.vmode == "qkv" else x0)
        return phi, s.softmax(-1) @ v


class Model(nn.Module):
    def __init__(self, cfg, d, d_in, K, L, n_cls=2):
        super().__init__()
        self.cfg, self.residual = cfg, cfg["residual"]
        self.Win = nn.Linear(d_in, d)
        self.enc = Encoder(d, cfg["enc"], cfg["agg"], cfg["readout"], K, L)
        self.att = GatedAttention(d, cfg["vmode"])
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_cls))

    def forward(self, x, b):
        h0 = torch.tanh(self.Win(x))                                        # x uniform => h0 uniform
        h = self.enc(h0, b)
        phi, o = self.att(h0, h, b.mask)
        z = o + phi if self.residual else o
        return self.head(z), dict(h=h, phi=phi, attn=o, attn_res=o + phi)


# ----------------------------------------------------------------------------------------
# 4. Part 1: pair distances (random initialisations, no training)
# ----------------------------------------------------------------------------------------
ENCODERS = {  # name: (kind, agg, readout)
    "V1 mean (paper default)": ("v1", "mean", None),
    "V1 sum  (1-GWL ref.)": ("v1", "sum", None),
    "V2 root, mean": ("v2", "mean", "root"),
    "V2 set-union, mean": ("v2", "mean", "set"),
    "V2 root, sum-inside (ext.)": ("v2", "sum", "root"),
}


def cfg_of(enc_name, vmode, residual=False):
    kind, agg, readout = ENCODERS[enc_name]
    return dict(enc=kind, agg=agg, readout=readout, vmode=vmode, residual=residual)


def pair_stats(out, b):
    """raw and normalised pair distances, split into 'differing |V_v|' and 'equal |V_v|' pairs."""
    res = {}
    a, c = b.pairs[:, 0], b.pairs[:, 1]
    for name, X in out.items():
        raw = (X[a] - X[c]).norm(dim=1)
        den = torch.zeros(len(raw))
        for gi, (off, n) in enumerate(b.slices):
            Xi = X[off:off + n]
            den[b.pair_gid == gi] = torch.cdist(Xi, Xi).sum() / (n * (n - 1))
        norm = torch.where(den > 1e-9, raw / den.clamp(min=1e-9), torch.zeros_like(raw))
        res[name] = (raw, norm)
    return res


def distance_experiment(insts, d, d_in, K, L, n_init, bs=16):
    batches = [collate(insts[i:i + bs]) for i in range(0, len(insts), bs)]
    rows = []
    for enc_name in ENCODERS:
        for vmode in ("qkv", "qk"):
            per_seed = defaultdict(list)
            for seed in range(n_init):
                torch.manual_seed(1000 + seed)
                model = Model(cfg_of(enc_name, vmode), d, d_in, K, L).eval()
                acc = defaultdict(lambda: defaultdict(list))
                with torch.no_grad():
                    for b in batches:
                        _, out = model(torch.ones(b.N, d_in), b)
                        for name, (raw, norm) in pair_stats(out, b).items():
                            dm = b.pair_diff
                            acc[name]["raw_diff"].append(raw[dm]); acc[name]["norm_diff"].append(norm[dm])
                            acc[name]["raw_eq"].append(raw[~dm])
                for name in ("h", "phi", "attn", "attn_res"):
                    rd, nd = torch.cat(acc[name]["raw_diff"]), torch.cat(acc[name]["norm_diff"])
                    per_seed[name + "_raw"].append(rd.median().item())
                    per_seed[name + "_norm"].append(nd.median().item())
                    per_seed[name + "_sep"].append((rd > 1e-9).double().mean().item())
                    re = torch.cat(acc[name]["raw_eq"])
                    per_seed[name + "_eq"].append(re.max().item() if len(re) else float("nan"))
            rows.append((enc_name, vmode, {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_seed.items()}))
    return rows


def fmt(x):
    return "0 (exact)" if x < 1e-12 else f"{x:.2e}" if x < 1e-3 else f"{x:.4f}"


def print_distance_table(rows):
    hdr = f"{'encoder':28s} {'attn':5s} | {'||h_v-h_u||':>12s} {'h norm.':>8s} {'||Phi||':>12s} {'||attn||':>12s} {'attn norm.':>11s} {'ratio':>6s} {'sep.%':>6s} | {'attn+res':>10s} {'sep.%':>6s}"
    print(hdr); print("-" * len(hdr))
    for enc, vm, s in rows:
        hn, an = s['h_norm'][0], s['attn_norm'][0]
        ratio = f"{an / hn:6.2f}" if hn > 1e-9 else "   n/a"
        print(f"{enc:28s} {vm.upper():5s} | {fmt(s['h_raw'][0]):>12s} {hn:8.3f} {fmt(s['phi_raw'][0]):>12s} "
              f"{fmt(s['attn_raw'][0]):>12s} {an:11.3f} {ratio} {100 * s['attn_sep'][0]:6.1f} | "
              f"{fmt(s['attn_res_raw'][0]):>10s} {100 * s['attn_res_sep'][0]:6.1f}")
    print("\nraw columns = median ||.|| over pairs with different |V_v|, mean over random inits.")
    print("norm. = median(||a_v-a_u|| / mean pairwise ||a_a-a_b|| in the instance);  ratio = attn norm. / h norm. (how much of the encoder's separation survives attention).")
    print("sep.% = share of pairs with distance > 1e-9.")
    print("Control (pairs with EQUAL |V_v|): max distance at encoder output, over inits:")
    for enc, vm, s in rows:
        if vm == "qkv":
            print(f"   {enc:28s} {fmt(s['h_eq'][0])}")


# ----------------------------------------------------------------------------------------
# 5. Part 2: structure-only classification
# ----------------------------------------------------------------------------------------
TRAIN_CFGS = [
    ("V1 mean + QKV (paper default)", cfg_of("V1 mean (paper default)", "qkv")),
    ("V1 sum  + QKV (1-GWL ref.)", cfg_of("V1 sum  (1-GWL ref.)", "qkv")),
    ("V2 root, mean + QKV", cfg_of("V2 root, mean", "qkv")),
    ("V2 set-union + QK-only", cfg_of("V2 set-union, mean", "qk")),
    ("V2 set-union + QK-only + residual", cfg_of("V2 set-union, mean", "qk", residual=True)),
    ("V2 set-union + QKV", cfg_of("V2 set-union, mean", "qkv")),
]


def plug_in_accuracy(train_feat, train_y, test_feat, test_y):
    """Best classifier that only sees a discrete feature: per-value majority label fit on train."""
    cnt = defaultdict(lambda: np.zeros(2))
    for f, y in zip(train_feat, train_y):
        cnt[f][y] += 1
    glob = np.bincount(train_y, minlength=2).argmax()
    pred = [cnt[f].argmax() if f in cnt else glob for f in test_feat]
    return float(np.mean(np.array(pred) == np.array(test_y)))


def train_experiment(train, val, test, args):
    d, d_in = args.d, 8
    tr = [collate(train[i:i + args.bs]) for i in range(0, len(train), args.bs)]
    va = [collate(val[i:i + args.bs]) for i in range(0, len(val), args.bs)]
    te = [collate(test[i:i + args.bs]) for i in range(0, len(test), args.bs)]

    def acc(model, batches):
        ok = n = 0
        with torch.no_grad():
            for b in batches:
                logits, _ = model(torch.ones(b.N, d_in), b)
                ok += (logits.argmax(1) == b.y).sum().item(); n += b.N
        return ok / n

    # reference classifiers (no neural net)
    ytr = torch.cat([I.y for I in train]).numpy(); yte = torch.cat([I.y for I in test]).numpy()
    allI = train + test
    cols = wl_colors(allI, rounds=8)
    ntr = sum(I.N for I in train)
    wl_flat = [c for col in cols for c in col]
    ball_flat = [int(x) for I in allI for x in I.ball]
    refs = {
        "majority class": float(max(yte.mean(), 1 - yte.mean())),
        "1-WL colour classifier (any MPNN)": plug_in_accuracy(wl_flat[:ntr], ytr, wl_flat[ntr:], yte),
        "|V_v| classifier (K-hop ball size)": plug_in_accuracy(ball_flat[:ntr], ytr, ball_flat[ntr:], yte),
    }
    print("reference accuracies on the test set (label depends on structure only):")
    for k, v in refs.items():
        print(f"   {k:38s} {100 * v:6.2f}")
    print()

    results = {}
    for name, cfg in TRAIN_CFGS:
        accs = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = Model(cfg, d, d_in, args.K, args.L)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_va, best_te, t0 = -1, 0, time.time()
            for ep in range(args.epochs):
                model.train()
                for i in np.random.RandomState(seed * 1000 + ep).permutation(len(tr)):
                    b = tr[i]
                    logits, _ = model(torch.ones(b.N, d_in), b)
                    loss = nn.functional.cross_entropy(logits, b.y)
                    opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                v = acc(model, va)
                if v > best_va:
                    best_va, best_te = v, acc(model, te)
            accs.append(best_te)
            print(f"   [{name}] seed {seed}: test {100 * best_te:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        results[name] = accs
    print(f"\n{'model (uniform features)':38s} test acc. % (mean +- std over seeds)")
    for name, a in results.items():
        print(f"{name:38s} {100 * np.mean(a):6.2f} +- {100 * np.std(a):.2f}")
    return refs, results


# ----------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="dist", choices=["dist", "train", "all"])
    p.add_argument("--n-inst", type=int, default=120)
    p.add_argument("--sizes", type=int, nargs="+", default=[12, 15, 18, 21], help="base-hypergraph node counts (n*d divisible by r)")
    p.add_argument("--deg", type=int, default=2); p.add_argument("--rank", type=int, default=3)
    p.add_argument("--K", type=int, default=2, help="BFS hops for V2 and hop count for V1")
    p.add_argument("--L", type=int, default=3, help="HGNN layers inside each sub-hypergraph (root readout needs L > K to see frontier effects)")
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--bs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # ---- Part 0: Appendix B sanity check
    ab = appendix_b_instance(args.K)
    wl = wl_colors([ab], 8)[0]
    print("== Part 0: Appendix-B pair (v in HG1, u in HG2) ==")
    print(f"   1-WL colours equal: {wl[0] == wl[8]}   |V_v|={int(ab.ball[0])} (paper: 7)   |V_u|={int(ab.ball[8])} (paper: 4)\n")

    # ---- dataset
    data = build_dataset(args.n_inst, args.sizes, args.deg, args.rank, args.K, args.seed)
    cols = wl_colors(data, 8)
    ok = all(cols[i][a.item()] == cols[i][b.item()] for i, I in enumerate(data) for a, b in I.pairs)
    n_pairs = sum(len(I.pairs) for I in data); n_diff = sum(int(I.pair_diff.sum()) for I in data)
    ylab = torch.cat([I.y for I in data]).double().mean().item()
    print(f"== Dataset: {len(data)} instances (base n in {args.sizes}, ({args.deg},{args.rank})-biregular, K={args.K}) ==")
    print(f"   all pairs 1-WL-equivalent: {ok}   pairs: {n_pairs}   with different |V_v|: {n_diff} ({100 * n_diff / n_pairs:.0f}%)")
    print(f"   label balance (share of 'acyclic K-hop ball'): {ylab:.2f}\n")
    assert ok

    if args.mode in ("dist", "all"):
        print("== Part 1: pair distances at random initialisation ==")
        rows = distance_experiment(data, args.d, 8, args.K, args.L, args.n_init, args.bs)
        print_distance_table(rows)
        print()
    if args.mode in ("train", "all"):
        print("== Part 2: structure-only node classification ==")
        n = len(data); a, b = int(0.6 * n), int(0.8 * n)
        train_experiment(data[:a], data[a:b], data[b:], args)


if __name__ == "__main__":
    main()