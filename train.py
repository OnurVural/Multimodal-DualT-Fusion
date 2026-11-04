"""Training and evaluation harness.

Trains one configuration on a training segment and evaluates it on the next,
over the requested segment pairs and seeds. Validation is held out on whole
active regions and is used for epoch selection and for the decision threshold;
the test segment is used once, for reporting.
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # python -I drops the script dir from sys.path
import framework as FW

CACHE = os.path.join(HERE, "cache")
CACHE_CROP = os.path.join(HERE, os.environ.get("CROP_CACHE", "cache_crop"))
# Both image sources use the same intensity scaling, so that comparing them
# isolates the crop rather than the contrast normalization.
CACHE_FULL = os.path.join(HERE, os.environ.get("FULL_CACHE", "cache_full"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
AR_RE = re.compile(r"_ar(\d+)_")


# --------------------------------------------------------------------------
def load_partition(part, image="crop", max_dt=None):
    ts = np.load(os.path.join(CACHE, "%s_ts.npy" % part))
    y = np.load(os.path.join(CACHE, "%s_y.npy" % part))
    names = json.load(open(os.path.join(CACHE, "%s_names.json" % part)))

    # Both image sources go through one path: align images to the time-series
    # order by instance name and apply the same pairing-tolerance filter, so the
    # crop and full-disk conditions differ only in the crop itself.
    src = CACHE_CROP if image == "crop" else CACHE_FULL
    img = np.load(os.path.join(src, "%s_img.npy" % part))
    cn_path = os.path.join(src, "%s_names.json" % part)
    dg_path = os.path.join(src, "%s_diag.json" % part)
    if not os.path.exists(cn_path):
        raise SystemExit(
            "%s lacks %s_names.json; build it with build_crop_cache.py so that "
            "image order and pairing offsets are recorded" % (src, part))
    cnames = json.load(open(cn_path))
    diag = json.load(open(dg_path)) if os.path.exists(dg_path) else [{}] * len(cnames)
    idx = {n: i for i, n in enumerate(cnames)}
    keep_src, keep_dst = [], []
    for i, n in enumerate(names):
        j = idx.get(n)
        if j is None:
            continue
        if max_dt is not None:
            dt = diag[j].get("dt_min", float("nan"))
            if not np.isfinite(dt) or dt > max_dt:
                continue
        keep_src.append(i); keep_dst.append(j)
    ts, y = ts[keep_src], y[keep_src]
    names = [names[i] for i in keep_src]
    img = img[keep_dst]

    ars = np.array([int(AR_RE.search(n).group(1)) if AR_RE.search(n) else -1 for n in names])
    return ts, img, y, names, ars


class MMDataset(Dataset):
    def __init__(self, ts, img, y):
        self.ts, self.img, self.y = ts, img, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        g = self.img[i].astype(np.float32) / 255.0
        im = (np.repeat(g[None], 3, axis=0) - MEAN) / STD
        return (torch.from_numpy(self.ts[i]), torch.from_numpy(im),
                torch.tensor(int(self.y[i]), dtype=torch.long))


def ar_disjoint_split(y, ars, val_frac=0.2, seed=0):
    """Hold out whole active regions, stratified by class."""
    rng = np.random.default_rng(seed)
    val_ars = set()
    for cls in (0, 1):
        a = np.unique(ars[y == cls])
        a = a[a >= 0]
        rng.shuffle(a)
        k = max(1, int(round(len(a) * val_frac)))
        val_ars.update(a[:k].tolist())
    is_val = np.array([a in val_ars for a in ars])
    # guard against a split that wipes out a class
    if len(np.unique(y[~is_val])) < 2 or len(np.unique(y[is_val])) < 2:
        idx = rng.permutation(len(y))
        is_val = np.zeros(len(y), bool)
        is_val[idx[: max(2, int(len(y) * val_frac))]] = True
    return ~is_val, is_val


# --------------------------------------------------------------------------
def all_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = (2.0 * (tp * tn - fn * fp) / denom) if denom else 0.0
    return dict(accuracy=float(accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
                tss=float(tpr - fpr), hss=float(hss),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn), threshold=float(thr))


def best_threshold(y, prob):
    cand = np.unique(np.round(prob, 3))
    if len(cand) > 400:
        cand = np.quantile(prob, np.linspace(0.01, 0.99, 400))
    best, bt = -2.0, 0.5
    for t in cand:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        tss = (tp / (tp + fn) if (tp + fn) else 0.0) - (fp / (fp + tn) if (fp + tn) else 0.0)
        if tss > best:
            best, bt = tss, float(t)
    return bt, best


@torch.no_grad()
def predict(model, ds, device, batch_size=32):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    ys, ps, al = [], [], []
    for ts, img, y in dl:
        out = model(ts.to(device), img.to(device))
        p = torch.softmax(out["logits"].float(), dim=1)[:, 1]
        ys.append(y.numpy()); ps.append(p.cpu().numpy())
        if out.get("alpha") is not None:
            al.append(out["alpha"].flatten().float().cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps),
            np.concatenate(al) if al else None)


def run_one(cfg_name, train_part, test_part, seed, args, device):
    build, use_supcon, image_src = FW.REGISTRY[cfg_name]
    if args.image_override:
        image_src = args.image_override

    ts, img, y, names, ars = load_partition(train_part, image_src, args.max_dt)
    tts, timg, ty, _, _ = load_partition(test_part, image_src, args.max_dt)
    tr_mask, va_mask = ar_disjoint_split(y, ars, args.val_frac, seed)

    d_tr = MMDataset(ts[tr_mask], img[tr_mask], y[tr_mask])
    d_va = MMDataset(ts[va_mask], img[va_mask], y[va_mask])
    d_te = MMDataset(tts, timg, ty)

    torch.manual_seed(seed); np.random.seed(seed)
    model = build().to(device)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    weighting = FW.LossWeighting().to(device)

    # the pretrained ViT wants a small lr; the from-scratch modules want a large one
    vit_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (vit_params if ".vit." in ("." + n + ".") or n.startswith("img.vit")
         or n.startswith("enc.vit") else other_params).append(p)
    other_params += [p for p in weighting.parameters() if p.requires_grad]
    groups = []
    if vit_params:
        groups.append(dict(params=vit_params, lr=args.lr))
    if other_params:
        groups.append(dict(params=other_params, lr=args.lr_head))
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)

    dl = DataLoader(d_tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    generator=torch.Generator().manual_seed(seed))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups],
        total_steps=max(1, args.epochs * len(dl)), pct_start=0.25)

    ce = nn.CrossEntropyLoss()

    best = dict(tss=-2.0, state=None, epoch=-1, thr=0.5)
    traj, t0 = [], time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for bts, bimg, by in dl:
            bts, bimg, by = bts.to(device), bimg.to(device), by.to(device)
            out = model(bts, bimg)
            l_ce = ce(out["logits"], by)
            if "logits_ts" in out:          # late fusion: supervise each branch
                l_ce = l_ce + ce(out["logits_ts"], by) + ce(out["logits_img"], by)
            l_con = (FW.supervised_contrastive_loss(out["z_ts"], out["z_img"], by,
                                                    temperature=args.supcon_temp)
                     if use_supcon else torch.zeros((), device=device))
            loss, wts = weighting(l_ce, l_con)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.item()); nb += 1

        vy, vp, _ = predict(model, d_va, device, args.eval_batch)
        vthr, vtss = best_threshold(vy, vp)
        vauc = roc_auc_score(vy, vp) if len(np.unique(vy)) > 1 else float("nan")
        traj.append(dict(epoch=ep + 1, loss=tot / max(nb, 1), val_tss=vtss,
                         val_auc=float(vauc), val_thr=vthr,
                         w_ce=float(wts[0]), w_con=float(wts[1])))
        if vtss > best["tss"]:
            best.update(tss=vtss, epoch=ep + 1, thr=vthr,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    train_min = (time.time() - t0) / 60.0

    ty_, tp_, alpha = predict(model, d_te, device, args.eval_batch)
    m_half = all_metrics(ty_, tp_, 0.5)
    m_val = all_metrics(ty_, tp_, best["thr"])

    t1 = time.time()
    predict(model, MMDataset(tts[:256], timg[:256], ty[:256]), device, args.eval_batch)
    latency_ms = (time.time() - t1) / max(1, min(256, len(ty))) * 1000.0

    res = dict(config=cfg_name, train=train_part, test=test_part, seed=seed,
               image=image_src, epochs=args.epochs,
               supcon=use_supcon, max_dt=args.max_dt, best_epoch=best["epoch"], val_tss=best["tss"],
               val_threshold=best["thr"],
               metrics_at_half=m_half, metrics_at_valthr=m_val,
               n_train=int(tr_mask.sum()), n_val=int(va_mask.sum()), n_test=len(ty),
               pos_rate_train=float(y[tr_mask].mean()), pos_rate_test=float(ty.mean()),
               n_train_ars=int(len(np.unique(ars[tr_mask]))),
               n_val_ars=int(len(np.unique(ars[va_mask]))),
               params=params, trainable=trainable, train_minutes=train_min,
               latency_ms_per_sample=latency_ms, trajectory=traj)
    if alpha is not None:
        res["alpha"] = dict(mean=float(alpha.mean()), std=float(alpha.std()),
                            min=float(alpha.min()), max=float(alpha.max()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pairs", nargs="+", default=["1-2", "2-3", "3-4", "4-5"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5, help="lr for the pretrained ViT")
    ap.add_argument("--lr-head", type=float, default=1e-3, help="lr for from-scratch modules")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--supcon-temp", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-dt", type=float, default=30.0)
    ap.add_argument("--image-override", default=None, choices=[None, "crop", "full"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results/main")
    args = ap.parse_args()

    out_dir = os.path.join(HERE, args.out)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    for cfg in args.configs:
        if cfg not in FW.REGISTRY:
            raise SystemExit("unknown config %r" % cfg)
    for cfg in args.configs:
        for seed in args.seeds:
            for pair in args.pairs:
                a, b = pair.split("-")
                tag = "%s_S%s-S%s_seed%d" % (cfg, a, b, seed)
                path = os.path.join(out_dir, "res_%s.json" % tag)
                if os.path.exists(path):
                    print("[skip] %s" % tag, flush=True); continue
                print("=== %s" % tag, flush=True)
                r = run_one(cfg, "partition%s" % a, "partition%s" % b, seed, args, device)
                with open(path, "w") as fh:
                    json.dump(r, fh, indent=2)
                mh, mv = r["metrics_at_half"], r["metrics_at_valthr"]
                print("  -> [0.5] acc=%.4f f1=%.4f auc=%.4f TSS=%.4f | [valthr=%.2f] "
                      "acc=%.4f f1=%.4f TSS=%.4f | ep%d %.1fmin"
                      % (mh["accuracy"], mh["f1"], mh["roc_auc"], mh["tss"],
                         r["val_threshold"], mv["accuracy"], mv["f1"], mv["tss"],
                         r["best_epoch"], r["train_minutes"]), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
