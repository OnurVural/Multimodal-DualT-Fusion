"""Aggregate the reported tables from the run records.

  results/main/   -- weight decay 0.1
  results/wd001/  -- weight decay 0.01

The proposed model is reported at the fixed recipe. Each baseline is reported
at whichever weight-decay value scores higher, and both are printed.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MET = ["accuracy", "precision", "recall", "f1", "roc_auc", "tss", "hss"]
SHORT = dict(accuracy="Acc", precision="Prec", recall="Rec", f1="F1",
             roc_auc="AUC", tss="TSS", hss="HSS")
PROPOSED = {"dualt"}

# the ungated residual sum was run under two gate modes that compute the same
# function, so their runs are pooled into one row
POOL = {"dualt_inert": "dualt_nogate"}

GROUPS = [
    ("trivial", ["Majority class"]),
    ("classical (time series only)",
     ["Logistic Reg [summary]", "Logistic Reg [flat]", "SVM (RBF) [summary]",
      "SVM (RBF) [flat]", "Random Forest [summary]", "Random Forest [flat]"]),
    ("unimodal", ["ts_lstm", "ts_transformer", "img_vit_fulldisk", "img_vit"]),
    ("alternative encoder architectures", ["dual_conv", "lstm_conv"]),
    ("multimodal fusion baselines",
     ["early_concat", "late_avg", "embed_concat"]),
    ("proposed", ["dualt"]),
    ("ablations of the proposed model",
     ["dualt_fulldisk", "dualt_nopos", "dualt_nosupcon", "dualt_nogate",
      "dualt_frozenvit"]),
]
PRETTY = {
    "Majority class": "Majority class (trivial)",
    "ts_lstm": "LSTM", "ts_transformer": "Transformer",
    "img_vit": "ViT, AR crop", "img_vit_fulldisk": "ViT, full disk",
    "early_concat": "Early fusion (raw concat)",
    "late_avg": "Late fusion (logit avg.)",
    "embed_concat": "Embedding concatenation",
    "dual_conv": "Dual convolutional", "lstm_conv": "LSTM + convolutional",
    "dualt": "DualT-Fusion (gated)",
    "dualt_nogate": "ungated residual sum",
    "dualt_nopos": "w/o positional encoding",
    "dualt_nosupcon": "w/o supervised contrastive loss",
    "dualt_frozenvit": "w/ frozen ViT",
    "dualt_fulldisk": "w/ full-disk images (no localization)",
}


def load(d):
    runs = defaultdict(list)
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "config" in r:
            runs[POOL.get(r["config"], r["config"])].append(r)
    return runs


def agg(rs, key):
    out = {}
    for m in MET:
        v = np.array([r[key][m] for r in rs if key in r and m in r[key]], float)
        v = v[np.isfinite(v)]
        out[m] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v)) if len(v) \
            else (np.nan, np.nan, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="metrics_at_valthr",
                    choices=["metrics_at_valthr", "metrics_at_half"])
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    A = load(os.path.join(HERE, "results", "main"))    # wd 0.1
    B = load(os.path.join(HERE, "results", "wd001"))   # wd 0.01

    chosen, note = {}, {}
    # classical/trivial baselines are not gradient-trained, so no wd arm applies
    def _nonneural(n):
        return "[" in n or n == "Majority class"
    for name in set(A) | set(B):
        if _nonneural(name):
            chosen[name], note[name] = (A.get(name) or B.get(name)), "n/a"
            continue
        if name in PROPOSED or name not in B or name not in A:
            src = A if name in A else B
            chosen[name] = src[name]
            note[name] = "wd0.1" if name in A and (name in PROPOSED or name not in B) else "wd0.01"
            continue
        ta = agg(A[name], args.key)["tss"][0]
        tb = agg(B[name], args.key)["tss"][0]
        if np.isfinite(tb) and (not np.isfinite(ta) or tb >= ta):
            chosen[name], note[name] = B[name], "wd0.01 (vs %.3f at wd0.1)" % ta
        else:
            chosen[name], note[name] = A[name], "wd0.1 (vs %.3f at wd0.01)" % tb

    print("threshold: %s" % ("validation-selected" if args.key == "metrics_at_valthr"
                             else "fixed 0.5"))
    print("=" * 124)
    print("%-36s %-4s " % ("method", "n") + "".join("%-14s" % SHORT[m] for m in MET)
          + " arm")
    print("-" * 124)
    for gname, names in GROUPS:
        rows = [n for n in names if n in chosen]
        if not rows:
            continue
        print("-- %s" % gname)
        for n in rows:
            a = agg(chosen[n], args.key)
            line = "%-36s %-4d " % (PRETTY.get(n, n), a["tss"][2])
            for m in MET:
                mu, sd, _ = a[m]
                line += "%-14s" % ("--" if not np.isfinite(mu)
                                   else "%.3f+-%.3f" % (mu, sd))
            print(line + " " + note.get(n, ""))

    missing = [n for _, ns in GROUPS for n in ns if n not in chosen]
    if missing:
        print("\nnot yet reported: %s" % ", ".join(missing))
    extra = sorted((set(chosen) - {n for _, ns in GROUPS for n in ns}))
    if extra:
        print("ungrouped configs present: %s" % ", ".join(extra))


if __name__ == "__main__":
    main()
