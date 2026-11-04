"""Representation figures for the trained model.

  fig_tsne.pdf       t-SNE of the time series, image and fused test embeddings
  fig_attention.pdf  ViT attention rollout over AR crops, with predictions

Trains DualT-Fusion on one segment pair, selects the epoch by validation TSS as
train.py does, and dumps both figures from the selected model.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import framework as FW
import train as T

P = argparse.Namespace(
    epochs=6, batch_size=16, eval_batch=32, lr=2e-5, lr_head=1e-3, wd=0.1,
    supcon_temp=0.1, w_con=0.1, loss_mode="uncertainty", class_weight=False,
    val_frac=0.2, max_dt=30.0, image_override=None)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TRAIN, TEST, SEED = "partition1", "partition2", 0
CKPT = os.path.join(HERE, "fig_model_dualt.pt")


def train():
    build, use_supcon, image_src = FW.REGISTRY["dualt"]
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu")
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = build().to(DEV)
        model.load_state_dict(ck["state"])
        tts, timg, ty, tnames, _ = T.load_partition(TEST, image_src, P.max_dt)
        print("reusing %s (epoch %d, thr %.3f)" % (CKPT, ck["epoch"], ck["thr"]), flush=True)
        return model, (tts, timg, ty, tnames), ck["thr"]
    ts, img, y, names, ars = T.load_partition(TRAIN, image_src, P.max_dt)
    tts, timg, ty, tnames, _ = T.load_partition(TEST, image_src, P.max_dt)
    tr, va = T.ar_disjoint_split(y, ars, P.val_frac, SEED)
    d_tr = T.MMDataset(ts[tr], img[tr], y[tr])
    d_va = T.MMDataset(ts[va], img[va], y[va])

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build().to(DEV)
    weighting = FW.LossWeighting(mode=P.loss_mode, w_ce=1.0, w_con=P.w_con).to(DEV)
    vit_p = [p for n, p in model.named_parameters()
             if p.requires_grad and n.startswith("img.vit")]
    oth_p = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("img.vit")]
    oth_p += list(weighting.parameters())
    opt = torch.optim.AdamW([dict(params=vit_p, lr=P.lr),
                             dict(params=oth_p, lr=P.lr_head)], weight_decay=P.wd)
    dl = DataLoader(d_tr, batch_size=P.batch_size, shuffle=True, drop_last=True,
                    generator=torch.Generator().manual_seed(SEED))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[P.lr, P.lr_head], total_steps=P.epochs * len(dl), pct_start=0.25)
    ce = nn.CrossEntropyLoss()
    best = dict(tss=-2.0, state=None, epoch=-1, thr=0.5)
    for ep in range(P.epochs):
        model.train()
        for bts, bimg, by in dl:
            bts, bimg, by = bts.to(DEV), bimg.to(DEV), by.to(DEV)
            out = model(bts, bimg)
            l_ce = ce(out["logits"], by)
            l_con = FW.supervised_contrastive_loss(out["z_ts"], out["z_img"], by,
                                                   temperature=P.supcon_temp)
            loss, _ = weighting(l_ce, l_con)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        vy, vp, _ = T.predict(model, d_va, DEV, P.eval_batch)
        thr, tss = T.best_threshold(vy, vp)
        print("epoch %d  val TSS %.4f  thr %.3f" % (ep + 1, tss, thr), flush=True)
        if tss > best["tss"]:
            best.update(tss=tss, epoch=ep + 1, thr=thr,
                        state={k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
    model.load_state_dict(best["state"])
    torch.save(dict(state=best["state"], thr=best["thr"], epoch=best["epoch"]), CKPT)
    print("selected epoch %d (val TSS %.4f)" % (best["epoch"], best["tss"]), flush=True)
    return model, (tts, timg, ty, tnames), best["thr"]


@torch.no_grad()
def embeddings(model, tts, timg, ty, n=1200):
    """Return the three embedding spaces the paper discusses."""
    model.eval()
    idx = np.random.RandomState(0).permutation(len(ty))[:n]
    dl = DataLoader(T.MMDataset(tts[idx], timg[idx], ty[idx]), batch_size=32)
    Z = {"ts": [], "img": [], "fused": []}
    for bts, bimg, _ in dl:
        bts, bimg = bts.to(DEV), bimg.to(DEV)
        z_ts = model.ts(bts)
        z_img = model.img(bimg)
        z, _ = model.fusion(z_ts, z_img)
        Z["ts"].append(z_ts.cpu().numpy())
        Z["img"].append(z_img.cpu().numpy())
        Z["fused"].append(z.cpu().numpy())
    return {k: np.concatenate(v) for k, v in Z.items()}, ty[idx]


def fig_tsne(Z, y):
    titles = [("ts", "(a) Time series embedding"),
              ("img", "(b) Image embedding"),
              ("fused", "(c) Fused embedding")]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
    for ax, (k, title) in zip(axes, titles):
        p = TSNE(n_components=2, init="pca", perplexity=30,
                 learning_rate="auto", random_state=0).fit_transform(Z[k])
        for lab, col, nm in [(0, "#3b7dd8", "Flare-negative"),
                             (1, "#e34a33", "Flare-positive")]:
            m = (y == lab)
            ax.scatter(p[m, 0], p[m, 1], s=3.2, c=col, alpha=0.55,
                       linewidths=0, label=nm)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="lower left", fontsize=7, markerscale=2.6,
                   frameon=True, framealpha=0.85)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(HERE, "fig_tsne." + ext),
                    dpi=180, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def rollout(vit, x):
    """Attention rollout (Abnar & Zuidema) over the CLS row, per sample."""
    out = vit(x, output_attentions=True)
    A = torch.stack(out.attentions)              # L x B x H x N x N
    A = A.mean(dim=2)                            # head average
    L, B, N, _ = A.shape
    eye = torch.eye(N, device=A.device).expand(B, N, N)
    R = eye.clone()
    for l in range(L):
        a = A[l] + eye
        a = a / a.sum(dim=-1, keepdim=True)
        R = a @ R
    cls = R[:, 0, 1:]                            # CLS attention over patches
    s = int(np.sqrt(cls.shape[1]))
    return cls.reshape(B, s, s).cpu().numpy()


@torch.no_grad()
def fig_attention(model, tts, timg, ty, thr, k=4):
    model.eval()
    # score a class-balanced random subset: the cache stores all flare-positive
    # instances before the negatives, so a leading slice would be single-class
    rs = np.random.RandomState(0)
    pos = rs.permutation(np.where(ty == 1)[0])[:200]
    neg = rs.permutation(np.where(ty == 0)[0])[:200]
    cand = np.concatenate([pos, neg])
    dl = DataLoader(T.MMDataset(tts[cand], timg[cand], ty[cand]), batch_size=32)
    probs = []
    for bts, bimg, _ in dl:
        o = model(bts.to(DEV), bimg.to(DEV))
        probs.append(torch.softmax(o["logits"].float(), 1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)
    pred = (probs >= thr).astype(int)
    yc = ty[cand]
    sel_local = []
    for lab in (1, 0):                       # one confident correct call each
        ok = [i for i in range(len(probs)) if yc[i] == lab and pred[i] == lab]
        ok.sort(key=lambda i: -abs(probs[i] - thr))
        sel_local += ok[:1]
    for lab in (1, 0):                       # one mistake of each kind, if any
        bad = [i for i in range(len(probs)) if yc[i] == lab and pred[i] != lab]
        if bad:
            bad.sort(key=lambda i: -abs(probs[i] - thr))
            sel_local.append(bad[0])
    sel = [int(cand[i]) for i in sel_local]
    probs = {int(cand[i]): float(probs[i]) for i in range(len(probs))}
    pred = {int(cand[i]): int(pred[i]) for i in range(len(pred))}
    ds = T.MMDataset(tts[sel], timg[sel], np.asarray(ty)[sel])
    x = torch.stack([ds[i][1] for i in range(len(sel))]).to(DEV)
    maps = rollout(model.img.vit, x)
    n = len(sel)
    fig, axes = plt.subplots(2, n, figsize=(1.75 * n, 3.7))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, i in enumerate(sel):
        im = timg[i]
        axes[0, j].imshow(im, cmap="gray")
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        m = maps[j]
        m = (m - m.min()) / (m.max() - m.min() + 1e-9)
        axes[1, j].imshow(im, cmap="gray")
        axes[1, j].imshow(np.kron(m, np.ones((im.shape[0] // m.shape[0],) * 2)),
                          cmap="inferno", alpha=0.5)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        truth = "FP" if ty[i] == 1 else "FN"
        got = "FP" if pred[i] == 1 else "FN"
        col = "#1a7f37" if pred[i] == ty[i] else "#d1242f"
        axes[0, j].set_title("true %s / pred %s\n$p=%.2f$" % (truth, got, probs[i]),
                             fontsize=7.5, color=col)
    axes[0, 0].set_ylabel("AR crop", fontsize=8)
    axes[1, 0].set_ylabel("Attention rollout", fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(HERE, "fig_attention." + ext),
                    dpi=180, bbox_inches="tight")
    plt.close()


def main():
    model, (tts, timg, ty, tnames), thr = train()
    Z, ys = embeddings(model, tts, timg, ty)
    # a quick quantitative handle on what the t-SNE panels show qualitatively
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    stats = {}
    for k in ("ts", "img", "fused"):
        stats[k] = float(cross_val_score(LogisticRegression(max_iter=2000),
                                         Z[k], ys, cv=5, scoring="roc_auc").mean())
    print("linear-probe AUC on test embeddings:", json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(os.path.join(HERE, "fig_probe_auc.json"), "w"), indent=1)
    fig_tsne(Z, ys)
    fig_attention(model, tts, timg, ty, thr)
    for f in ("fig_tsne.pdf", "fig_attention.pdf"):
        p = os.path.join(HERE, f)
        print("wrote %s (%.0f KB)" % (p, os.path.getsize(p) / 1024), flush=True)


if __name__ == "__main__":
    main()
