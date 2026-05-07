import numpy as np
import torch



def rand_train_test_idx(label, train_prop, valid_prop, balance=False):
    if not balance:
        n = label.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.randperm(n)

        train_idx = perm[:train_num]
        valid_idx = perm[train_num:train_num + valid_num]
        test_idx = perm[train_num + valid_num:]

        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }

    else:
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([ind[:percls_trn] for ind in indices], dim=0)
        rest_index = torch.cat([ind[percls_trn:] for ind in indices], dim=0)
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]

        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }

    return split_idx