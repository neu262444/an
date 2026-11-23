import torch

from features import label_trick_x, train_label_onehot


def test_label_trick_never_exposes_non_training_labels():
    torch.manual_seed(0)
    N, D = 50, 10
    y = torch.randint(0, 4, (N,))
    train_idx = torch.randperm(N)[:20]
    others = torch.ones(N, dtype=torch.bool)
    others[train_idx] = False
    oh = train_label_onehot(y, train_idx, D)
    assert oh[others].abs().sum() == 0                                   # no label outside the training set
    assert torch.equal(oh[train_idx].argmax(1), y[train_idx])
    assert torch.all(oh[train_idx].sum(1) == 1)
    noise = torch.randn(N, D)
    inp = train_idx[:10]
    x = label_trick_x(noise, oh, inp)
    changed = (x != noise).any(1).nonzero().flatten()
    assert torch.equal(changed.sort().values, inp.sort().values)         # only the input nodes carry labels
    assert torch.equal(x[others], noise[others])
