# usage: python paired_test.py HSAT_attn HSAT_mlp cora citeseer ...
#        python paired_test.py HSAT_attn EDGNN    cora citeseer ...
import json, sys
import numpy as np
from scipy import stats

RATIO = 0.25 / 0.5   # n_test / n_train for the 50/25/25 split
TA, TB = sys.argv[1], sys.argv[2]

for d in sys.argv[3:]:
    A = json.load(open(f"results/{d}_{TA}.json"))
    B = json.load(open(f"results/{d}_{TB}.json"))
    assert A["splits"] == B["splits"], f"{d}: splits differ, comparison is not paired"
    a, b = np.array(A["test"]), np.array(B["test"])
    diff, k = a - b, len(a)
    if np.all(diff == 0):
        print(f"{d:15s} identical on every run"); continue
    t_p = stats.ttest_rel(a, b).pvalue
    w_p = stats.wilcoxon(a, b).pvalue
    t_c = diff.mean() / np.sqrt((1 / k + RATIO) * diff.var(ddof=1))   # Nadeau-Bengio correction
    c_p = 2 * stats.t.sf(abs(t_c), df=k - 1)
    print(f"{d:15s} {TA} {a.mean():.2f}±{a.std(ddof=1):.2f} | {TB} {b.mean():.2f}±{b.std(ddof=1):.2f} | "
          f"diff {diff.mean():+.2f}±{diff.std(ddof=1):.2f} | {TA} wins {np.sum(diff > 0)}/{k} | "
          f"paired t p={t_p:.3f} | Wilcoxon p={w_p:.3f} | corrected t p={c_p:.3f}")
