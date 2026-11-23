import numpy as np

from features import incidence, spectral_embedding, structural_features


def _toy(seed=0, n=80, m=60):
    rng = np.random.default_rng(seed)
    V, E = [], []
    for e in range(m):
        nodes = rng.choice(n, rng.integers(2, 6), replace=False)
        V += nodes.tolist()
        E += [e] * len(nodes)
    return np.array(V), np.array(E), n, m


def test_spectral_matches_dense_and_svd():
    V, E, n, _ = _toy()
    B = incidence(V, E, n)
    dv = np.maximum(np.asarray(B.sum(1)).ravel(), 1)
    de = np.maximum(np.asarray(B.sum(0)).ravel(), 1)
    Bn = np.diag(dv ** -.5) @ B.toarray() @ np.diag(de ** -.5)
    w, U = np.linalg.eigh(Bn @ Bn.T)
    k = 12
    vals, vecs = spectral_embedding(B, k)
    assert np.allclose(vals, w[::-1][:k], atol=1e-8)
    assert np.allclose(np.linalg.norm(U[:, ::-1][:, :k].T @ vecs, axis=0), 1, atol=1e-6)
    assert np.allclose(vals, np.linalg.svd(Bn, compute_uv=False)[:k] ** 2, atol=1e-8)


def test_features_standardized_and_deterministic():
    V, E, n, _ = _toy()
    X = structural_features(V, E, n, 20)
    assert X.shape == (n, 20) and np.isfinite(X).all()
    sd = X.std(0)
    assert np.allclose(X.mean(0), 0, atol=1e-8)
    assert np.all((np.abs(sd - 1) < 1e-8) | (sd < 1e-12))
    assert np.allclose(X, structural_features(V, E, n, 20))


def test_self_loops_do_not_change_features():
    V, E, n, m = _toy()
    X = structural_features(V, E, n, 20)
    V2 = np.concatenate([V, np.arange(n)])
    E2 = np.concatenate([E, np.arange(m, m + n)])
    assert np.allclose(X, structural_features(V2, E2, n, 20))
