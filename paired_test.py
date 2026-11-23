# usage: python paired_test.py TAG_A TAG_B dataset1 dataset2 ...
#   tags as written by train.py, e.g. HSAT_attn, EDGNN, HSAT_attn_tr0.1_va0.1
import json, sys
import numpy as np
from scipy import stats

TA, TB = sys.argv[1], sys.argv[2]
rows = []
for d in sys.argv[3:]:
    A = json.load(open(f"results/{d}_{TA}.json"))
    B = json.load(open(f"results/{d}_{TB}.json"))
    assert A["splits"] == B["splits"], f"{d}: splits differ, comparison is not paired"
    pa = (A.get("train_prop", 0.5), A.get("valid_prop", 0.25))
    pb = (B.get("train_prop", 0.5), B.get("valid_prop", 0.25))
    assert pa == pb, f"{d}: different split proportions {pa} vs {pb}"
    ratio = (1 - pa[0] - pa[1]) / pa[0]            # n_test / n_train
    a, b = np.array(A["test"], dtype=float), np.array(B["test"], dtype=float)
    diff, k = a - b, len(a)
    m, sd = diff.mean(), diff.std(ddof=1)
    tq = stats.t.ppf(0.975, df=k - 1)
    half = tq * sd / np.sqrt(k)                     # paired 95% t-interval
    half_c = tq * sd * np.sqrt(1 / k + ratio)       # Nadeau-Bengio corrected (splits overlap)
    w_p = stats.wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
    rows.append(dict(d=d, a=a, b=b, m=m, ci=(m - half, m + half), cic=(m - half_c, m + half_c),
                     wins=int(np.sum(diff > 0)), k=k, w_p=w_p))

# Holm correction of the Wilcoxon p-values across the datasets on the command line
running = 0.0
for rank, i in enumerate(np.argsort([r["w_p"] for r in rows])):
    running = max(running, min(1.0, (len(rows) - rank) * rows[i]["w_p"]))
    rows[i]["w_holm"] = running

print(f"{'dataset':15s} {TA:>14s} {TB:>14s} {'gap':>6s} {'95% CI':>16s} {'corrected 95% CI':>18s} "
      f"{'wins':>6s} {'Wilcoxon p':>10s} {'Holm p':>7s}")
for r in rows:
    print(f"{r['d']:15s} {r['a'].mean():8.2f}±{r['a'].std(ddof=1):5.2f} {r['b'].mean():8.2f}±{r['b'].std(ddof=1):5.2f} "
          f"{r['m']:+6.2f} [{r['ci'][0]:+6.2f}, {r['ci'][1]:+6.2f}] [{r['cic'][0]:+7.2f}, {r['cic'][1]:+7.2f}] "
          f"{r['wins']:>3d}/{r['k']:<2d} {r['w_p']:10.4f} {r['w_holm']:7.4f}")
