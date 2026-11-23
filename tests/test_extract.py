import torch

from datasets import extract_sub_hypergraphs

AB = [[0, 1, 2], [1, 3, 4], [4, 5, 6], [2, 7, 6], [8, 9, 10], [9, 10, 11], [12, 13, 14], [13, 14, 15]]


def _ei(edges):
    V = [v for e in edges for v in e]
    E = [j for j, e in enumerate(edges) for _ in e]
    return torch.tensor([V, E])


def _node_sets(sub, n):
    _, sub_root, sub_idx = sub
    sets = [set() for _ in range(n)]
    for r, u in zip(sub_root.tolist(), sub_idx.tolist()):
        sets[r].add(u)
    return sets


def _random_edges(seed=0, n=60, m=50):
    g = torch.Generator().manual_seed(seed)
    return [torch.randperm(n, generator=g)[:int(torch.randint(2, 6, (1,), generator=g))].tolist()
            for _ in range(m)], n


def test_uncapped_matches_appendix_b():
    sets = _node_sets(extract_sub_hypergraphs(_ei(AB), 16, k_hop=2), 16)
    assert len(sets[0]) == 7 and len(sets[8]) == 4


def test_large_cap_equals_uncapped():
    edges, n = _random_edges()
    a = extract_sub_hypergraphs(_ei(edges), n, k_hop=2)
    b = extract_sub_hypergraphs(_ei(edges), n, k_hop=2, max_edges=1000)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_cap_is_subset_and_deterministic():
    edges, n = _random_edges()
    full = _node_sets(extract_sub_hypergraphs(_ei(edges), n, k_hop=2), n)
    c1 = extract_sub_hypergraphs(_ei(edges), n, k_hop=2, max_edges=1, seed=3)
    c2 = extract_sub_hypergraphs(_ei(edges), n, k_hop=2, max_edges=1, seed=3)
    assert all(torch.equal(x, y) for x, y in zip(c1, c2))
    capped = _node_sets(c1, n)
    assert all(c <= f for c, f in zip(capped, full))
    assert sum(map(len, capped)) < sum(map(len, full))
