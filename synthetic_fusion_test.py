#!/usr/bin/env python3
"""
Q/K vs Q/K/V on synthetic hypergraphs: where does structure have to enter the attention block?

Self-contained (no import of the V1/V2 script). ONE fixed structural encoder for every arm, so this
file is only about the fusion location, never about the encoder variant.

Arms (same parameters, same initialisation stream, same training recipe; only the wiring differs)

    id    Q, K from   V from    meaning
    x     h0          h0        X-only: plain Transformer, no structure
    qk    Phi         h0        QK-struct: structure only in the attention weights (SAT-style, Eq. 5)
    v     h0          Phi       V-struct: structure only in the payload
    qkv   Phi         Phi       QKV: HSA-attn (Eq. 6)

    h0 = tanh(W_in x)      h = encoder(.)      Phi = sigmoid(psi1 h0) * h0 + sigmoid(psi2 h) * h   (Eq. 9)

Stripped model: one gated block, single-head exponential-kernel attention, MLP head. No dropout, no
normalisation, no positional encodings, and by default no skip connection (--skips none). A skip is
architecture, not optimisation, so it is a switch: --skips none raw phi  (raw adds h0, phi adds Phi).

Parts (--parts)
    exp1  UNTRAINED, uniform features. Share of 1-WL-equivalent node pairs whose block outputs differ.
          Checks Theorem 1 numerically: (i) exact collapse without structural V, (ii) separation with it.
    exp2  TRAINED, label = structure bit only. Features are x = 1 + spread * noise, for each value in
          --spreads. spread 0 is the uniform-feature regime of Theorem 1; spread > 0 gives features that
          differ between nodes but say nothing about the label.
    exp3  TRAINED, interaction task: label = feature bit XOR structure bit. Neither signal alone beats 50%.
    exp4  TRAINED sweeps (--sweeps):
            lambda : label = structure bit with probability lambda, else feature bit
            signal : label = structure bit, and the features carry the label with strength a (--signals)

Structure bit s_v = "K-hop sub-hypergraph of v is Berge-acyclic" (same generator as the V1/V2 script:
biregular base hypergraph + its 2-lift, all paired nodes 1-WL-equivalent). Feature bit f_v ~ Bernoulli(1/2),
independent of the structure. Features: x_v = 1 + a * (2 bit_v - 1) * u + sigma * eps_v, u = (+1,..,-1,..).

Evaluation nodes: by default each instance contributes equally many s = 1 and s = 0 nodes (the larger
class is subsampled; every node still takes part in attention). Chance is then 50% in every part and no
arm can profit from the base rate of an instance. --no-balance evaluates on all nodes instead.

Reference classifiers (no neural net) are printed with every trained part so the numbers can be read honestly:
    majority | logistic probe on raw x | 1-WL colour + feature bit (ceiling for any message-passing encoder)
    | |V_v| + feature bit (ceiling for THIS encoder) | per-instance oracle given the feature bit
    | frozen random encoder + logistic probe -> s (does h carry the structure bit at all?)

What is guaranteed and what is not
  * Uniform features (exp1, exp2 with spread 0): for ANY weights, x and qk give one output for all nodes
    and v gives one output per instance. All three sit exactly at chance. This is Theorem 1(i).
  * Non-uniform features: nothing forces qk to chance. Its output is a convex combination of raw-feature
    values, and the structure-aware weights can still move it inside that hull, so it can carry structure
    whenever the features differ between nodes, even if they are pure noise. Likewise v can use noisy
    features as node identifiers, attend to itself and read its own Phi. Whether training finds these
    routes is exactly what exp2 (spread > 0), exp3 and exp4 measure. Do not pre-commit to the outcome.
  * Noise-free two-valued features (--xor-noise 0): the outputs of x and v depend on a node only through
    its feature bit, so both are bounded by the per-instance oracle. qk is still not bounded.

Encoder input (--enc-input)
    const (default)  the encoder runs on a learned constant, so h_v depends on structure only. This is the
                     setting of Theorem 1 (h_w = phi(w, H) next to x_w) and keeps the two signals separate.
    x                the encoder runs on h0, as in the paper's pipeline; h_v then also mixes in features.
  With uniform features the two coincide up to the value of the constant. With --enc-input x --no-balance
  --skips none, arms qk / qkv of exp2 (spread 0) use the same weights and data as (a0) / (c0) of the earlier
  Q/K vs Q/K/V script.

Cost: one "setting" trains 4 arms x --seeds x --epochs. Defaults give 3 (exp2) + 2 (exp3) + 10 (exp4)
settings. For a first pass use e.g.  --seeds 3 --epochs 30.
"""
import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

D_IN = 8
U_PM = np.array([1.0] * (D_IN // 2) + [-1.0] * (D_IN - D_IN // 2))   # direction that carries the feature bit
ENC_SPEC = ("v2", "mean", "set")   # the ONE encoder used everywhere: sub-hypergraph extractor, set-union readout
ENC_NAME = "V2 (set-union readout)"

ARMS = {  # id: (label, Q/K source, V source)
    "x": ("X-only     Q,K,V from x", "raw", "raw"),
    "qk": ("QK-struct  Q,K from Phi, V from x    [SAT-style]", "phi", "raw"),
    "v": ("V-struct   Q,K from x,   V from Phi", "raw", "phi"),
    "qkv": ("QKV        Q,K,V from Phi            [HSA-attn]", "phi", "phi"),
}
SHORT = {"x": "X-only", "qk": "QK-struct", "v": "V-struct", "qkv": "QKV"}


# ------------------------------------------------------------------ data generation (as in the V1/V2 script)
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
        self.ball, self.y = torch.tensor(ball), torch.tensor(acyc)   # y = structure bit s
        self.pairs = torch.tensor(pairs)
        self.pair_diff = torch.tensor([ball[a] != ball[b] for a, b in pairs])


def make_cover_instance(n, d, r, K, rng):
    base = biregular(n, d, r, rng)
    lift = two_lift(n, base, rng)
    edges = base + [[v + n for v in e] for e in lift]
    pairs = [(i, n + i + n * s) for i in range(n) for s in (0, 1)]
    return Instance(3 * n, edges, pairs, K)


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


# ------------------------------------------------------------------ tasks: features, labels, evaluation nodes
def make_task(insts, kind, sep=1.0, noise=0.0, lam=1.0, balance=True, seed=0):
    """Attach x (features), lab (label), fb (feature bit), sel (evaluation nodes) to copies of the instances.

    kind    feature bit carried by x     label
    struct  none (x = 1 + noise*eps)     s
    xor     f                            f XOR s
    mix     f                            s with probability lam, else f
    signal  s (strength sep)             s
    The random draws are made in the same order for every kind, so f, eps and sel are shared across settings.
    """
    rng = np.random.RandomState(seed)
    out = []
    for I in insts:
        n, s = I.N, I.y.numpy()
        f = rng.randint(0, 2, n)
        eps = rng.randn(n, D_IN)
        pick = rng.rand(n) < lam
        perm = rng.permutation(n)
        if kind == "struct":
            bit, lab, fb = None, s, np.zeros(n, dtype=np.int64)
        elif kind == "xor":
            bit, lab, fb = f, f ^ s, f
        elif kind == "mix":
            bit, lab, fb = f, np.where(pick, s, f), f
        elif kind == "signal":
            bit, lab, fb = s, s, None
        else:
            raise ValueError(kind)
        x = 1.0 + noise * eps
        if bit is not None:
            x = x + sep * (2 * bit - 1)[:, None] * U_PM[None, :]
        if fb is None:                                   # signal task: the bit a reader of x would decode
            fb = ((x - 1.0) @ U_PM > 0).astype(np.int64)
        sel = np.ones(n, dtype=bool)
        if balance:
            m = int(min(s.sum(), n - s.sum()))
            sel[:] = False
            for c in (0, 1):
                sel[perm[s[perm] == c][:m]] = True
        J = copy.copy(I)
        J.x, J.sel = torch.tensor(x), torch.tensor(sel)
        J.lab, J.fb = torch.tensor(np.asarray(lab, dtype=np.int64)), torch.tensor(np.asarray(fb, dtype=np.int64))
        out.append(J)
    return out


class Batch:
    pass


def collate(insts):
    b, N, M, NS, MS = Batch(), 0, 0, 0, 0
    cols = defaultdict(list)
    for gi, I in enumerate(insts):
        cols["vi"].append(I.vi + N); cols["ei"].append(I.ei + M)
        cols["svi"].append(I.svi + NS); cols["sei"].append(I.sei + MS)
        cols["node_map"].append(I.node_map + N); cols["root_id"].append(I.root_id + N)
        cols["gid"].append(torch.full((I.N,), gi)); cols["s"].append(I.y); cols["ball"].append(I.ball)
        cols["pairs"].append(I.pairs + N); cols["pair_diff"].append(I.pair_diff)
        cols["x"].append(I.x); cols["lab"].append(I.lab); cols["fb"].append(I.fb); cols["sel"].append(I.sel)
        N, M, NS, MS = N + I.N, M + I.M, NS + I.n_sub, MS + I.m_sub
    for k, v in cols.items():
        setattr(b, k, torch.cat(v))
    b.N, b.M, b.n_sub, b.m_sub = N, M, NS, MS
    b.mask = b.gid[:, None] == b.gid[None, :]            # attention only inside an instance
    return b


def batches_of(insts, bs):
    return [collate(insts[i:i + bs]) for i in range(0, len(insts), bs)]


# ------------------------------------------------------------------ encoder (Eqs. 10-13), fixed for all arms
class HGNN(nn.Module):
    """Scatter-gather, Eqs. (10)-(11), mean aggregation (W2, W4 = identity)."""

    def __init__(self, d, agg):
        super().__init__()
        self.agg = agg
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
        if self.readout == "set":
            return torch.zeros(b.N, hs.size(1)).index_add(0, b.root_id, hs)
        is_root = b.node_map == b.root_id
        return torch.zeros(b.N, hs.size(1)).index_copy(0, b.root_id[is_root], hs[is_root])


# ------------------------------------------------------------------ the block and the four arms
class Block(nn.Module):
    """Eq. (9) gate + single-head exponential-kernel attention with selectable Q/K source, V source and skip.
    All parameters are created for every arm (unused ones simply receive no gradient), so the four arms
    start from the same initialisation for a given seed."""

    def __init__(self, d, qk_src, v_src, skip):
        super().__init__()
        self.d, self.qk_src, self.v_src, self.skip = d, qk_src, v_src, skip
        self.needs_phi = "phi" in (qk_src, v_src, skip)
        self.psi1, self.psi2 = nn.Linear(d, d), nn.Linear(d, d)
        self.Wq, self.Wk, self.Wv = (nn.Linear(d, d, bias=False) for _ in range(3))

    def gates(self, h0, h):
        return torch.sigmoid(self.psi1(h0)), torch.sigmoid(self.psi2(h))

    def forward(self, h0, h, mask):
        phi = None
        if self.needs_phi:
            g1, g2 = self.gates(h0, h)
            phi = g1 * h0 + g2 * h                                                    # Eq. (9)
        qk = phi if self.qk_src == "phi" else h0
        s = (self.Wq(qk) @ self.Wk(qk).T / math.sqrt(self.d)).masked_fill(~mask, float("-inf"))
        o = s.softmax(-1) @ self.Wv(phi if self.v_src == "phi" else h0)
        if self.skip == "phi":
            o = o + phi
        elif self.skip == "raw":
            o = o + h0
        return o


class Model(nn.Module):
    def __init__(self, d, K, L, arm, skip, enc_input):
        super().__init__()
        _, qk_src, v_src = ARMS[arm]
        kind, agg, readout = ENC_SPEC
        self.enc_input = enc_input
        self.Win = nn.Linear(D_IN, d)
        self.enc = Encoder(d, kind, agg, readout, K, L)
        self.att = Block(d, qk_src, v_src, skip)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))       # nonlinear: XOR needs it
        self.Ws = nn.Linear(D_IN, d)   # structural-channel input for --enc-input const (created last on purpose)

    def embed(self, b, force_enc=False):
        h0 = torch.tanh(self.Win(b.x))
        h = None
        if self.att.needs_phi or force_enc:
            e_in = h0 if self.enc_input == "x" else torch.tanh(self.Ws(torch.ones(b.N, D_IN)))
            h = self.enc(e_in, b)
        return h0, h

    def repr(self, b):
        h0, h = self.embed(b)
        return self.att(h0, h, b.mask)

    def forward(self, b):
        return self.head(self.repr(b))


# ------------------------------------------------------------------ reference classifiers
def plug_in_accuracy(train_feat, train_y, test_feat, test_y):
    cnt = defaultdict(lambda: np.zeros(2))
    for f, y in zip(train_feat, train_y):
        cnt[f][y] += 1
    glob = np.bincount(train_y, minlength=2).argmax()
    pred = [cnt[f].argmax() if f in cnt else glob for f in test_feat]
    return float(np.mean(np.array(pred) == np.array(test_y)))


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


def node_table(insts, cols):
    """One row per evaluation node: (instance id, label, feature bit, ball size, 1-WL colour)."""
    rows = []
    for gi, (I, col) in enumerate(zip(insts, cols)):
        lab, fb, ball = I.lab.numpy(), I.fb.numpy(), I.ball.numpy()
        for v in np.nonzero(I.sel.numpy())[0]:
            rows.append((gi, int(lab[v]), int(fb[v]), int(ball[v]), col[v]))
    return rows


def reference_accuracies(train, test, tr_b, te_b, args):
    cols = wl_colors(train + test, 8)
    tr, te = node_table(train, cols[:len(train)]), node_table(test, cols[len(train):])
    ytr, yte = [r[1] for r in tr], [r[1] for r in te]
    refs = {"majority class": float(max(np.mean(yte), 1 - np.mean(yte)))}

    Xtr, Ytr = torch.cat([b.x[b.sel] for b in tr_b]), torch.cat([b.lab[b.sel] for b in tr_b])
    Xte, Yte = torch.cat([b.x[b.sel] for b in te_b]), torch.cat([b.lab[b.sel] for b in te_b])
    refs["logistic probe on raw features x"] = logistic_probe(Xtr, Ytr, Xte, Yte)
    refs["1-WL colour + feature bit (any MPNN encoder)"] = plug_in_accuracy(
        [(r[4], r[2]) for r in tr], ytr, [(r[4], r[2]) for r in te], yte)
    refs["|V_v| + feature bit (ceiling, this encoder)"] = plug_in_accuracy(
        [(r[3], r[2]) for r in tr], ytr, [(r[3], r[2]) for r in te], yte)

    cnt = defaultdict(lambda: np.zeros(2))          # uses the test labels of each instance: an upper bound
    for gi, y, fb, _, _ in te:
        cnt[(gi, fb)][y] += 1
    refs["per-instance oracle given feature bit (bound: X-only, V-struct if x is two-valued)"] = \
        float(sum(c.max() for c in cnt.values()) / max(len(te), 1))

    accs = []                                        # does the frozen, untrained encoder expose s linearly?
    for seed in range(min(args.seeds, 5)):
        torch.manual_seed(seed)
        model = Model(args.d, args.K, args.L, "qkv", "none", args.enc_input).eval()
        with torch.no_grad():
            Htr = torch.cat([model.embed(b, True)[1][b.sel] for b in tr_b])
            Hte = torch.cat([model.embed(b, True)[1][b.sel] for b in te_b])
        accs.append(logistic_probe(Htr, torch.cat([b.s[b.sel] for b in tr_b]),
                                   Hte, torch.cat([b.s[b.sel] for b in te_b])))
    refs["frozen encoder h + logistic probe -> structure bit s"] = float(np.mean(accs))
    return refs


def print_refs(refs):
    print("reference classifiers on the test evaluation nodes:")
    for k, v in refs.items():
        print(f"   {k:84s} {100 * v:6.2f}")
    print()


# ------------------------------------------------------------------ training
def accuracy(model, batches):
    ok = n = 0
    with torch.no_grad():
        for b in batches:
            logits = model(b)
            pred = (logits[:, 1] - logits[:, 0] > 1e-9).long()
            # pred = model(b).argmax(1)
            ok += (pred[b.sel] == b.lab[b.sel]).sum().item()
            n += int(b.sel.sum().item())
    return ok / max(n, 1)


def gate_means(model, batches):
    """Mean feature-gate and structure-gate activation (over nodes and dimensions), as in Fig. 5."""
    if not model.att.needs_phi:
        return float("nan"), float("nan")
    g1s = g2s = n = 0
    with torch.no_grad():
        for b in batches:
            h0, h = model.embed(b)
            g1, g2 = model.att.gates(h0, h)
            g1s, g2s, n = g1s + g1.mean(1).sum().item(), g2s + g2.mean(1).sum().item(), n + b.N
    return g1s / n, g2s / n


def train_one(arm, skip, tr, va, te, args, seed):
    torch.manual_seed(seed)
    model = Model(args.d, args.K, args.L, arm, skip, args.enc_input)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_va, best_te = -1.0, 0.0
    for ep in range(args.epochs):
        model.train()
        for i in np.random.RandomState(seed * 1000 + ep).permutation(len(tr)):
            b = tr[i]
            if not b.sel.any():
                continue
            loss = F.cross_entropy(model(b)[b.sel], b.lab[b.sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        v = accuracy(model, va)
        if v > best_va:
            best_va, best_te = v, accuracy(model, te)
    return best_te, gate_means(model, va)


def arm_row(arm, skip, r, args):
    a = np.array(r["acc"])
    line = (f"{ARMS[arm][0]:52s} {skip:5s} {100 * a.mean():17.2f} +- {100 * a.std():5.2f}  "
            f"{100 * a.min():6.2f} {100 * a.max():6.2f}   ({r['sec']:.0f}s)")
    if args.gates and not np.isnan(r["gates"][0][0]):
        g = np.array(r["gates"]).mean(0)
        line += f"   gates: feat {g[0]:.3f}  struct {g[1]:.3f}"
    return line


def run_setting(splits, args, kind, show=True, **task_kw):
    """Build the task on the three splits, compute references, train every arm x skip x seed.
    show=True prints the references and one row per arm as soon as it is finished."""
    tr_i, va_i, te_i = (make_task(x, kind, balance=args.balance, seed=args.seed + 1 + k, **task_kw)
                        for k, x in enumerate(splits))
    tr, va, te = (batches_of(x, args.bs) for x in (tr_i, va_i, te_i))
    refs = reference_accuracies(tr_i, te_i, tr, te, args)
    if show:
        print_refs(refs)
        print(f"{'arm':52s} {'skip':5s} {'test acc. % (mean +- std)':>26s}  {'min':>6s} {'max':>6s}", flush=True)
    res = {}
    for skip in args.skips:
        for arm in args.arms:
            t0, accs, gates = time.time(), [], []
            for seed in range(args.seeds):
                a, g = train_one(arm, skip, tr, va, te, args, seed)
                accs.append(a)
                gates.append(g)
            res[f"{arm}|{skip}"] = {"acc": accs, "gates": gates, "sec": time.time() - t0}
            if show:
                print(arm_row(arm, skip, res[f"{arm}|{skip}"], args), flush=True)
    if show:
        print()
    return refs, res


# ------------------------------------------------------------------ Experiment 1: untrained collapse / injectivity
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


def exp1(data, args):
    task = make_task(data[:args.dist_inst], "struct", noise=0.0, balance=False, seed=args.seed + 1)
    batches = batches_of(task, args.bs)
    print(f"== Experiment 1 (UNTRAINED, uniform features): block outputs of 1-WL-equivalent node pairs "
          f"({len(task)} instances, {args.seeds} random inits) ==")
    print(f"{'arm':52s} {'skip':5s} {'separated, |V_v| differs':>26s} {'|V_v| equal':>12s} "
          f"{'median norm. dist.':>20s} {'corr. with ||h_v-h_v`||':>24s}")
    out = {}
    for skip in args.skips:
        for arm in args.arms:
            sd, se, nm, cr = [], [], [], []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                model = Model(args.d, args.K, args.L, arm, skip, args.enc_input).eval()
                raws, ns, dms, hds = [], [], [], []
                with torch.no_grad():
                    for b in batches:
                        raw, nrm = norm_pair_dist(model.repr(b), b)
                        h = model.embed(b, True)[1]
                        raws.append(raw); ns.append(nrm); dms.append(b.pair_diff)
                        hds.append((h[b.pairs[:, 0]] - h[b.pairs[:, 1]]).norm(dim=1))
                raw, nrm, dm, hd = (torch.cat(z) for z in (raws, ns, dms, hds))
                sd.append(100 * (raw[dm] > 1e-9).double().mean().item())
                se.append(100 * (raw[~dm] > 1e-9).double().mean().item() if (~dm).any() else float("nan"))
                nm.append(nrm[dm].median().item())
                r, g = raw[dm].numpy(), hd[dm].numpy()
                cr.append(float(np.corrcoef(r, g)[0, 1]) if r.std() > 1e-12 and g.std() > 1e-12 else float("nan"))
            corr = "n/a (constant)" if np.all(np.isnan(cr)) else f"{np.nanmean(cr):.3f}"
            print(f"{ARMS[arm][0]:52s} {skip:5s} {np.mean(sd):17.1f} +- {np.std(sd):3.1f}% {np.nanmean(se):11.1f}% "
                  f"{np.mean(nm):12.3f} +- {np.std(nm):.3f} {corr:>24s}")
            out[f"{arm}|{skip}"] = {"sep_diff": sd, "sep_equal": se, "norm_dist": nm, "corr": cr}
    print("   separated = output distance > 1e-9; exactly 0.0% means the two outputs are identical.\n"
          "   '|V_v| equal' pairs are a control (the encoder cannot tell them apart; expected 0 without a skip).\n")
    return out


# ------------------------------------------------------------------ Experiment 2: structure-only label
def exp2(splits, args):
    out = {}
    for spread in args.spreads:
        what = "UNIFORM features (Theorem 1 regime)" if spread == 0 else f"features = 1 + {spread:g} * noise (uninformative)"
        print(f"== Experiment 2: label = structure bit, {what} | {args.seeds} seeds, {args.epochs} epochs ==")
        refs, res = run_setting(splits, args, "struct", noise=spread)
        out[f"spread={spread:g}"] = {"refs": refs, "res": res}
    return out


# ------------------------------------------------------------------ Experiment 3: interaction (XOR) task
def exp3(splits, args):
    out = {}
    for noise in args.xor_noise:
        what = "noise-free two-valued features" if noise == 0 else f"feature noise sd {noise:g}"
        print(f"== Experiment 3: label = feature bit XOR structure bit, separation {args.feat_sep:g}, {what} "
              f"| {args.seeds} seeds, {args.epochs} epochs ==")
        refs, res = run_setting(splits, args, "xor", sep=args.feat_sep, noise=noise)
        out[f"noise={noise:g}"] = {"refs": refs, "res": res}
    return out


# ------------------------------------------------------------------ Experiment 4: sweeps
CEIL = "|V_v| + feature bit (ceiling, this encoder)"
XPROBE = "logistic probe on raw features x"


def exp4(splits, args):
    out = {}
    for sweep in args.sweeps:
        if sweep == "lambda":
            vals, name = args.lambdas, "lambda"
            print(f"== Experiment 4a: label = structure bit w.p. lambda, else feature bit "
                  f"(separation {args.feat_sep:g}, feature noise sd {args.feat_noise:g}) ==")
        else:
            vals, name = args.signals, "signal a"
            print(f"== Experiment 4b: label = structure bit; features carry the label with strength a "
                  f"(feature noise sd {args.feat_noise:g}) ==")
        rows = []
        for val in vals:
            if sweep == "lambda":
                refs, res = run_setting(splits, args, "mix", False, sep=args.feat_sep, noise=args.feat_noise, lam=val)
            else:
                refs, res = run_setting(splits, args, "signal", False, sep=val, noise=args.feat_noise)
            rows.append((val, refs, res))
            print(f"   done {name} = {val:g}", flush=True)
        for skip in args.skips:
            print(f"\nskip = {skip}   (test acc. %, mean +- std over {args.seeds} seeds)")
            print(f"{name:>9s} {'ceiling':>8s} {'x-probe':>8s} | " + " ".join(f"{SHORT[a]:>15s}" for a in args.arms)
                  + ("   QKV gates feat/struct" if args.gates and "qkv" in args.arms else ""))
            for val, refs, res in rows:
                cells = []
                for arm in args.arms:
                    a = np.array(res[f"{arm}|{skip}"]["acc"])
                    cells.append(f"{100 * a.mean():7.2f} +- {100 * a.std():4.2f}")
                line = f"{val:9.3g} {100 * refs[CEIL]:8.2f} {100 * refs[XPROBE]:8.2f} | " + " ".join(cells)
                if args.gates and "qkv" in args.arms:
                    g = np.array(res[f"qkv|{skip}"]["gates"]).mean(0)
                    line += f"   {g[0]:.3f} / {g[1]:.3f}"
                print(line)
            if args.plot:
                plot_sweep(rows, skip, name, f"{args.plot}_{sweep}_{skip}.png", args)
        print()
        out[sweep] = [{"value": v, "refs": r, "res": s} for v, r, s in rows]
    return out


def plot_sweep(rows, skip, name, path, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   (matplotlib not available, skipping plot)")
        return
    xs = [v for v, _, _ in rows]
    plt.figure(figsize=(4.6, 3.4))
    for arm in args.arms:
        a = np.array([s[f"{arm}|{skip}"]["acc"] for _, _, s in rows]) * 100
        plt.errorbar(xs, a.mean(1), yerr=a.std(1), marker="o", capsize=2, label=SHORT[arm])
    plt.plot(xs, [100 * r[CEIL] for _, r, _ in rows], "k--", lw=1, label="ceiling")
    plt.xlabel(name); plt.ylabel("test acc. (%)"); plt.legend(fontsize=7); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    print(f"   saved {path}")


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", default=["exp1", "exp2", "exp3", "exp4"],
                   choices=["exp1", "exp2", "exp3", "exp4"])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--skips", nargs="+", default=["none"], choices=["none", "raw", "phi"],
                   help="skip around the attention operator: none (stripped model), raw (adds h0), phi (adds Phi)")
    p.add_argument("--enc-input", default="const", choices=["const", "x"],
                   help="const: h depends on structure only (Theorem 1 setting); x: encoder runs on features")
    p.add_argument("--no-balance", dest="balance", action="store_false",
                   help="evaluate on all nodes instead of a per-instance balanced subset of the structure bit")
    # tasks
    p.add_argument("--spreads", type=float, nargs="+", default=[0.0, 0.1, 1.0], help="exp2: feature noise sd")
    p.add_argument("--xor-noise", type=float, nargs="+", default=[0.0, 0.5], help="exp3: feature noise sd")
    p.add_argument("--sweeps", nargs="+", default=["lambda", "signal"], choices=["lambda", "signal"])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--signals", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4])
    p.add_argument("--feat-sep", type=float, default=1.0, help="exp3 / exp4a: feature-bit separation a")
    p.add_argument("--feat-noise", type=float, default=0.5, help="exp4: feature noise sd")
    p.add_argument("--gates", action="store_true", help="also report mean gate activations (Fig. 5 quantities)")
    # data (same generator and defaults as the V1/V2 script)
    p.add_argument("--n-inst", type=int, default=600)
    p.add_argument("--dist-inst", type=int, default=200)
    p.add_argument("--sizes", type=int, nargs="+", default=[12, 15, 18, 21, 24, 27, 30])
    p.add_argument("--deg", type=int, default=2)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--L", type=int, default=3)
    # model / training
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0, help="dataset / task seed")
    p.add_argument("--out", default=None, help="write all results to this JSON file")
    p.add_argument("--plot", default=None, help="exp4: save one PNG per sweep with this filename prefix")
    args = p.parse_args()

    data = build_dataset(args.n_inst, args.sizes, args.deg, args.rank, args.K, args.seed)
    cols = wl_colors(data, 8)
    assert all(cols[i][a.item()] == cols[i][c.item()] for i, I in enumerate(data) for a, c in I.pairs), \
        "pairs are not 1-WL-equivalent -- generator bug"
    n_pairs = sum(len(I.pairs) for I in data)
    n_diff = sum(int(I.pair_diff.sum()) for I in data)
    share = torch.cat([I.y for I in data]).double().mean().item()
    print(f"== Dataset: {len(data)} instances, base n in {args.sizes}, ({args.deg},{args.rank})-biregular, K={args.K} ==")
    print(f"   all pairs 1-WL-equivalent: True   |V_v| differs for {n_diff}/{n_pairs} pairs ({100 * n_diff / n_pairs:.0f}%)")
    print(f"   share of nodes with structure bit 1 (acyclic K-hop ball): {share:.2f}")
    print(f"   encoder: {ENC_NAME}, input: {args.enc_input}   evaluation nodes: "
          f"{'per-instance balanced on the structure bit' if args.balance else 'all nodes'}\n")

    n = len(data)
    a, c = int(0.6 * n), int(0.8 * n)
    splits = (data[:a], data[a:c], data[c:])
    results = {"args": vars(args)}
    if "exp1" in args.parts:
        results["exp1"] = exp1(data, args)
    if "exp2" in args.parts:
        results["exp2"] = exp2(splits, args)
    if "exp3" in args.parts:
        results["exp3"] = exp3(splits, args)
    if "exp4" in args.parts:
        results["exp4"] = exp4(splits, args)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"results written to {args.out}")


if __name__ == "__main__":
    main()