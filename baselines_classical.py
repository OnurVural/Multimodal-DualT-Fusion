"""Classical and trivial baselines, under the same protocol as the deep models:
the same pairing-tolerance filter, the same active-region-disjoint validation
split, a decision threshold chosen on validation, and the same metric code.

Two feature constructions are reported:
  summary : per-parameter mean/std/min/max/first/last  (144 dims)
  flat    : the 60x24 z-scored window flattened        (1440 dims)
"""
import argparse, json, os, sys, time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import load_partition, ar_disjoint_split, best_threshold, all_metrics

PAIRS = ["1-2", "2-3", "3-4", "4-5"]


def feats(ts, kind):
    if kind == "flat":
        return ts.reshape(len(ts), -1)
    return np.concatenate([ts.mean(1), ts.std(1), ts.min(1), ts.max(1),
                           ts[:, 0, :], ts[:, -1, :]], axis=1)


def models(seed, balanced):
    cw = "balanced" if balanced else None
    return {
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight=cw, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight=cw,
                                                random_state=seed, n_jobs=-1),
        "Logistic Reg": LogisticRegression(max_iter=3000, class_weight=cw, random_state=seed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-dt", type=float, default=30.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--out", default="results/main")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, args.out)
    os.makedirs(out_dir, exist_ok=True)

    for seed in args.seeds:
        for pair in PAIRS:
            a, b = pair.split("-")
            tr_p, te_p = "partition%s" % a, "partition%s" % b
            # images are irrelevant to these models; use the crop cache so the
            # dt filter drops exactly the same samples as the deep runs
            ts, _, y, _, ars = load_partition(tr_p, "crop", args.max_dt)
            tts, _, ty, _, _ = load_partition(te_p, "crop", args.max_dt)
            tr_m, va_m = ar_disjoint_split(y, ars, args.val_frac, seed)

            for kind in ["summary", "flat"]:
                Xtr, Xva, Xte = (feats(ts[tr_m], kind), feats(ts[va_m], kind),
                                 feats(tts, kind))
                ytr, yva = y[tr_m], y[va_m]

                todo = dict(models(seed, args.class_weight))
                for name, mdl in todo.items():
                    tag = "%s [%s]_S%s-S%s_seed%d" % (name, kind, a, b, seed)
                    path = os.path.join(out_dir, "cls_%s.json" % tag.replace(" ", "_").replace("/", "-"))
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    mdl.fit(Xtr, ytr)
                    vp = mdl.predict_proba(Xva)[:, 1]
                    thr, vtss = best_threshold(yva, vp)
                    tp_ = mdl.predict_proba(Xte)[:, 1]
                    res = dict(config="%s [%s]" % (name, kind), train=tr_p, test=te_p,
                               seed=seed, image="n/a", max_dt=args.max_dt,
                               val_tss=vtss, val_threshold=thr,
                               metrics_at_half=all_metrics(ty, tp_, 0.5),
                               metrics_at_valthr=all_metrics(ty, tp_, thr),
                               n_train=int(tr_m.sum()), n_val=int(va_m.sum()),
                               n_test=len(ty), train_minutes=(time.time() - t0) / 60.0,
                               params=0, trainable=0)
                    with open(path, "w") as fh:
                        json.dump(res, fh, indent=2)
                    print("  %-34s TSS@0.5=%.4f  TSS@valthr=%.4f  (%.1fs)"
                          % (tag, res["metrics_at_half"]["tss"],
                             res["metrics_at_valthr"]["tss"],
                             res["train_minutes"] * 60), flush=True)

            # trivial reference, once per pair/seed
            tag = "Majority_class_S%s-S%s_seed%d" % (a, b, seed)
            path = os.path.join(out_dir, "cls_%s.json" % tag)
            if not os.path.exists(path):
                maj = int(np.bincount(y[tr_m]).argmax())
                prob = np.full(len(ty), float(maj))
                res = dict(config="Majority class", train=tr_p, test=te_p, seed=seed,
                           image="n/a", max_dt=args.max_dt, val_tss=0.0, val_threshold=0.5,
                           metrics_at_half=all_metrics(ty, prob, 0.5),
                           metrics_at_valthr=all_metrics(ty, prob, 0.5),
                           n_train=int(tr_m.sum()), n_val=int(va_m.sum()),
                           n_test=len(ty), train_minutes=0.0, params=0, trainable=0)
                with open(path, "w") as fh:
                    json.dump(res, fh, indent=2)
                print("  %-34s acc=%.4f TSS=%.4f" % (tag, res["metrics_at_half"]["accuracy"],
                                                     res["metrics_at_half"]["tss"]), flush=True)
    print("CLASSICAL DONE", flush=True)


if __name__ == "__main__":
    main()
