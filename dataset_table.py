"""Per-partition dataset composition, as actually constructed after filtering.

Reports instance counts, class balance, and the number of distinct active
regions contributing to each class.
"""
import os, sys
os.environ.setdefault("CROP_CACHE", "cache_crop")
os.environ.setdefault("FULL_CACHE", "cache_full")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from train import load_partition

rows = []
for p in range(1, 6):
    part = "partition%d" % p
    _, _, y, names, ars = load_partition(part, "crop", 30.0)
    pos, neg = y == 1, y == 0
    ar_pos = set(ars[pos].tolist()) - {-1}
    ar_neg = set(ars[neg].tolist()) - {-1}
    rows.append(dict(
        part=p, n=len(y), npos=int(pos.sum()), nneg=int(neg.sum()),
        rate=float(pos.mean()), arpos=len(ar_pos), arneg=len(ar_neg),
        wpos=pos.sum() / max(len(ar_pos), 1)))

tot_n = sum(r["n"] for r in rows)
tot_p = sum(r["npos"] for r in rows)

print("%-6s %7s %7s %7s %7s %7s %7s %8s" % (
    "part", "N", "pos", "neg", "pos%", "ARpos", "ARneg", "win/AR"))
for r in rows:
    print("%-6d %7d %7d %7d %6.1f%% %7d %7d %8.1f" % (
        r["part"], r["n"], r["npos"], r["nneg"], 100 * r["rate"],
        r["arpos"], r["arneg"], r["wpos"]))
print("%-6s %7d %7d %7d %6.1f%%" % (
    "all", tot_n, tot_p, tot_n - tot_p, 100 * tot_p / tot_n))

print("\n--- LaTeX ---")
print(r"\begin{tabular}{lrrrrrr}")
print(r"\toprule")
print(r"Partition & Instances & Positive & Negative & Positive rate & "
      r"\multicolumn{2}{c}{Distinct ARs} \\")
print(r"\cmidrule(lr){6-7}")
print(r" & & & & & Positive & Negative \\")
print(r"\midrule")
for r in rows:
    print(r"$S_%d$ & %d & %d & %d & %.1f\%% & %d & %d \\" % (
        r["part"], r["n"], r["npos"], r["nneg"], 100 * r["rate"],
        r["arpos"], r["arneg"]))
print(r"\midrule")
print(r"All & %d & %d & %d & %.1f\%% & -- & -- \\" % (
    tot_n, tot_p, tot_n - tot_p, 100 * tot_p / tot_n))
print(r"\bottomrule")
print(r"\end{tabular}")
