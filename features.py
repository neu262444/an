"""Label-free node features for the label-independent-feature experiment (reviewer C1Qs W3).

structural : 4 degree statistics + (feat_dim - 4) spectral coordinates: the top eigenvectors of
             A = Dv^-1/2 B De^-1 B^T Dv^-1/2, i.e. the left singular vectors of the normalized incidence
             matrix Dv^-1/2 B De^-1/2 (= the smallest eigenvectors of the hypergraph Laplacian I - A).
random     : fixed N(0, 1) features (a floor: label-free and structure-free).
label_trick: fixed N(0, 1) noise plus, for training nodes only, their one-hot label (see train.py); never uses
             validation or test labels.
Both use only the hypergraph, never labels. Singleton hyperedges are ignored, so the self-loops that
train.py adds for most methods do not change the features. Columns are standardized over all nodes,
and the result is cached, so every method and every run sees identical features.
Delete the cached .npy file to recompute.
"""
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
import torch


def incidence(V, E, n):
    """Binary node x hyperedge incidence matrix without singleton hyperedges."""
    B = sp.coo_matrix((np.ones(len(V)), (V, E)), shape=(n, int(E.max()) + 1)).tocsr()
    B.sum_duplicates()
    B.data[:] = 1.0
    size = np.asarray(B.sum(0)).ravel()
    return B[:, np.where(size >= 2)[0]].tocsr()


def spectral_embedding(B, k, seed=0):
    """Top-k eigenpairs of A = Dv^-1/2 B De^-1 B^T Dv^-1/2, without forming A."""
    n = B.shape[0]
    assert k < n, f"need feat_dim - 4 < number of nodes ({k} >= {n})"
    dv = np.maximum(np.asarray(B.sum(1)).ravel(), 1)
    de = np.maximum(np.asarray(B.sum(0)).ravel(), 1)
    Bn = (sp.diags(1 / np.sqrt(dv)) @ B @ sp.diags(1 / np.sqrt(de))).tocsr()
    BnT = Bn.T.tocsr()
    op = LinearOperator((n, n), matvec=lambda x: Bn @ (BnT @ x), dtype=np.float64)
    v0 = np.random.default_rng(seed).standard_normal(n)
    vals, vecs = eigsh(op, k=k, which="LA", v0=v0)
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    vecs = vecs * np.sign(vecs[np.abs(vecs).argmax(0), np.arange(k)])   # fix the sign of each vector
    return vals, vecs


def standardize(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def structural_features(V, E, n, dim):
    B = incidence(V, E, n)
    dv = np.asarray(B.sum(1)).ravel()
    de = np.asarray(B.sum(0)).ravel()
    Bc = B.tocoo()
    max_size = np.zeros(n)
    np.maximum.at(max_size, Bc.row, de[Bc.col])
    mean_size = (B @ de) / np.maximum(dv, 1)
    n_nbrs = np.maximum((B @ B.T).getnnz(axis=1) - (dv > 0), 0)          # distinct co-members
    stats = np.stack([np.log1p(dv), np.log1p(mean_size), np.log1p(max_size), np.log1p(n_nbrs)], 1)
    vals, vecs = spectral_embedding(B, dim - stats.shape[1])
    print(f"[features] structural: {B.shape[1]} hyperedges of size >= 2, {vecs.shape[1]} spectral dims, "
          f"eigenvalues {vals[0]:.4f} .. {vals[-1]:.4f} "
          f"({int((vals > 1 - 1e-6).sum())} equal to 1 = connected-component indicators)")
    return standardize(np.hstack([stats, vecs]))


def random_features(n, dim, seed):
    return np.random.default_rng(seed).standard_normal((n, dim))


def label_free_features(data, args):
    """Return an [N, feat_dim] float32 array for args.feat_source in {'structural', 'random'}."""
    name = f"feat_{args.feat_source}_d{args.feat_dim}"
    if args.feat_source == "random":
        name += f"_s{args.feat_seed}"
    path = os.path.join(args.data_dir, "processed", name + ".npy")
    if os.path.exists(path):
        X = np.load(path)
        print(f"[features] loaded {path}")
    else:
        n = int(data.num_nodes)
        if args.feat_source == "structural":
            V = data.edge_index[0].cpu().numpy()
            E = data.edge_index[1].cpu().numpy()
            X = structural_features(V, E, n, args.feat_dim)
        elif args.feat_source == "random":
            X = random_features(n, args.feat_dim, args.feat_seed)
        else:
            raise ValueError(args.feat_source)
        X = X.astype(np.float32)
        np.save(path, X)
        print(f"[features] saved {path}")
    assert X.shape == (int(data.num_nodes), args.feat_dim), (X.shape, data.num_nodes)
    return X.astype(np.float32)


def train_label_onehot(y, train_idx, dim):
    """One-hot labels of the TRAINING nodes only (all other rows stay zero), padded to `dim` columns."""
    out = torch.zeros(y.shape[0], dim, device=y.device)
    out[train_idx, y[train_idx]] = 1.0
    return out


def label_trick_x(noise, train_onehot, input_idx):
    """Noise for every node, plus the one-hot label for the nodes in input_idx (a subset of the training nodes)."""
    x = noise.clone()
    x[input_idx] += train_onehot[input_idx]
    return x
