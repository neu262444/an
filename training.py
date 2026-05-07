import copy
import operator
from enum import Enum, auto
from typing import List
import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class StopVariable(Enum):
    LOSS = auto()
    ACCURACY = auto()
    NONE = auto()

class Best(Enum):
    RANKED = auto()
    ALL = auto()

def Stop_args(patience = 100, max_epochs=2000):
    return dict(stop_varnames=[StopVariable.ACCURACY, StopVariable.LOSS], patience=patience, max_epochs=max_epochs, remember=Best.RANKED)

class EarlyStopping:
    def __init__(
            self, model: Module, stop_varnames: List[StopVariable],
            patience: int = 10, max_epochs: int = 200, remember: Best = Best.ALL):
        self.model = model
        self.comp_ops = []
        self.stop_vars = []
        self.best_vals = []
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                self.stop_vars.append('loss')
                self.comp_ops.append(operator.le)
                self.best_vals.append(np.inf)
            elif stop_varname is StopVariable.ACCURACY:
                self.stop_vars.append('acc')
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
        self.remember = remember
        self.remembered_vals = copy.copy(self.best_vals)
        self.max_patience = patience
        self.patience = self.max_patience
        self.max_epochs = max_epochs
        self.best_epoch = None
        self.best_state = None

    def check(self, values: List[np.floating], epoch: int) -> bool:
        checks = [self.comp_ops[i](val, self.best_vals[i])
                  for i, val in enumerate(values)]
        if any(checks):
            self.best_vals = np.choose(checks, [self.best_vals, values])
            self.patience = self.max_patience

            comp_remembered = [
                    self.comp_ops[i](val, self.remembered_vals[i])
                    for i, val in enumerate(values)]
            if self.remember is Best.ALL:
                if all(comp_remembered):
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values)
                    self.best_state = {
                            key: value.cpu() for key, value
                            in self.model.state_dict().items()}
            elif self.remember is Best.RANKED:
                for i, comp in enumerate(comp_remembered):
                    if comp:
                        if not(self.remembered_vals[i] == values[i]):
                            self.best_epoch = epoch
                            self.remembered_vals = copy.copy(values)
                            self.best_state = {
                                    key: value.cpu() for key, value
                                    in self.model.state_dict().items()}
                            break
                    else:
                        break
        else:
            self.patience -= 1
        return self.patience == 0
    


class Logger:
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, log_path=None):
        self.log_path = log_path
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, train_acc, valid_acc, test_acc):
        result = [train_acc, valid_acc, test_acc]
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            max_train = result[:, 0].max().item()
            max_test = result[:, 2].max().item()

            argmax = result[:, 1].argmax().item()
            train = result[argmax, 0].item()
            valid = result[argmax, 1].item()
            test = result[argmax, 2].item()
            return {'max_train': max_train, 'max_test': max_test,
                'train': train, 'valid': valid, 'test': test}
        else:
            keys = ['max_train', 'max_test', 'train', 'valid', 'test']

            best_results = []
            for r in range(len(self.results)):
                best_results.append([self.get_statistics(r)[k] for k in keys])

            ret_dict = {}
            best_result = torch.tensor(best_results)
            for i, k in enumerate(keys):
                ret_dict[k+'_mean'] = best_result[:, i].mean().item()
                ret_dict[k+'_std'] = best_result[:, i].std().item()

            return ret_dict

    def print_statistics(self, run=None):
        if run is not None:
            result = self.get_statistics(run)
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result['max_train']:.2f}")
            print(f"Highest Valid: {result['valid']:.2f}")
            print(f"  Final Train: {result['train']:.2f}")
            print(f"   Final Test: {result['test']:.2f}")
        else:
            result = self.get_statistics()
            print(f"All runs:")
            print(f"Highest Train: {result['max_train_mean']:.2f} ± {result['max_train_std']:.2f}")
            print(f"Highest Valid: {result['valid_mean']:.2f} ± {result['valid_std']:.2f}")
            print(f"  Final Train: {result['train_mean']:.2f} ± {result['train_std']:.2f}")
            print(f"   Final Test: {result['test_mean']:.2f} ± {result['test_std']:.2f}")

    def final(self):
        result = self.get_statistics()
        return result['test_mean']
    
class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False

class NodeClsEvaluator:

    def __init__(self):
        return

    def eval(self, y_true, y_pred):

        acc_list = []
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

        is_labeled = (~np.isnan(y_true)) & (~np.isinf(y_true)) # no nan and inf
        correct = (y_true[is_labeled] == y_pred[is_labeled])
        acc_list.append(float(np.sum(correct))/len(correct))

        missclassified = y_true[is_labeled] != y_pred[is_labeled]

        return {'acc': sum(correct) / sum(is_labeled), 'missclassified': missclassified}




def setup_training_components(model, args):
    """
    Initializes optimizer, learning rate scheduler, and early stopping handler.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    total_updates = getattr(args, 'tot_updates', args.epochs)
    warmup_updates = getattr(args, 'warmup_updates', int(total_updates * 0.1))

    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=warmup_updates,
        tot_updates=total_updates,
        lr=args.lr,
        end_lr=getattr(args, 'end_lr', 0.0001),
        power=1.0,
    )

    patience = getattr(args, 'patience', int(args.epochs * 0.15))
    stopping_args = Stop_args(patience=patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
          
    return optimizer, lr_scheduler, early_stopping
