"""Encoders, fusion, objectives and the model registry for DualT-Fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from transformers import ViTModel

D = 128
VIT_NAME = "google/vit-base-patch16-224-in21k"


# --------------------------------------------------------------------------
# time series encoders
# --------------------------------------------------------------------------
class TSTransformer(nn.Module):
    def __init__(self, input_dim=24, embed_dim=D, num_heads=4, num_layers=2,
                 dropout=0.1, ff_dim=256, seq_len=60, positional=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.positional = positional
        if positional:
            self.pos = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                           dim_feedforward=ff_dim, dropout=dropout,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        h = self.input_proj(x)
        if self.positional:
            h = h + self.pos[:, : h.size(1)]
        h = self.transformer(h)
        pooled = h.mean(dim=1)
        return Fn.normalize(self.proj(pooled), dim=1)


class TSLSTM(nn.Module):
    def __init__(self, input_dim=24, hidden=D, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout)
        self.proj = nn.Linear(hidden, D)

    def forward(self, x):
        h, _ = self.lstm(x)
        return Fn.normalize(self.proj(h[:, -1]), dim=1)


class TSConv(nn.Module):
    def __init__(self, input_dim=24, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.ReLU())
        self.proj = nn.Linear(hidden, D)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))          # (B, C, T)
        return Fn.normalize(self.proj(h.mean(dim=2)), dim=1)


# --------------------------------------------------------------------------
# image encoders
# --------------------------------------------------------------------------
class ImgViT(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        self.vit = ViTModel.from_pretrained(VIT_NAME)
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
        self.frozen = freeze
        hid = self.vit.config.hidden_size
        self.proj_head = nn.Sequential(nn.Linear(hid, 256), nn.ReLU(), nn.Linear(256, D))

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                out = self.vit(x)
        else:
            out = self.vit(x)
        return Fn.normalize(self.proj_head(out.pooler_output), dim=1)


class ImgConv(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.net = nn.Sequential(*blk(3, 32), *blk(32, 64), *blk(64, 128), *blk(128, 256))
        self.proj = nn.Linear(256, D)

    def forward(self, x):
        h = self.net(x)                                  # (B, 256, h, w)
        return Fn.normalize(self.proj(h.mean(dim=(2, 3))), dim=1)


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------
class ScalarGateFusion(nn.Module):
    """Gated residual merge of the two pooled modality embeddings.

    z = q + alpha * v, with q, k, v linear projections of the normalized
    embeddings and alpha a single scalar per sample. The mode selects how
    alpha is produced:

      'sigmoid' : alpha = sigmoid(q.k / tau + b), the gate used by DualT-Fusion
      'none'    : alpha = 1, an ungated residual sum
      'inert'   : softmax over a singleton axis, which is identically 1 and
                  therefore also an ungated residual sum
    """

    def __init__(self, embed_dim=D, mode="sigmoid", gate_init_bias=2.0):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.log_tau = nn.Parameter(torch.zeros(1))   # tau = exp(log_tau) > 0
        self.bias = nn.Parameter(torch.tensor(float(gate_init_bias)))
        self.mode = mode

    def forward(self, z_ts, z_img):
        q, k, v = self.W_q(z_ts), self.W_k(z_img), self.W_v(z_img)
        if self.mode == "none":
            return q + v, torch.ones(q.size(0), 1, device=q.device)
        s = (q * k).sum(dim=-1, keepdim=True)
        logits = s / (torch.exp(self.log_tau) + 1e-6)
        if self.mode == "inert":
            alpha = torch.softmax(logits, dim=1)
        else:
            alpha = torch.sigmoid(logits + self.bias)
        return q + alpha * v, alpha


# --------------------------------------------------------------------------
# objectives
# --------------------------------------------------------------------------
def supervised_contrastive_loss(z1, z2, labels, temperature=0.1):
    """Supervised contrastive loss over both modality embeddings of a batch.

    The two embedding sets are concatenated into 2B anchors sharing the batch
    labels; positives are the other anchors of the same class.
    """
    z = torch.cat([Fn.normalize(z1, dim=1), Fn.normalize(z2, dim=1)], dim=0)
    y = torch.cat([labels, labels], dim=0)
    sim = (z @ z.t()) / temperature
    n = z.size(0)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y.unsqueeze(1) == y.unsqueeze(0)) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    cnt = pos.sum(1)
    valid = cnt > 0
    if valid.sum() == 0:
        return z.sum() * 0.0
    loss = -(log_prob.masked_fill(~pos, 0.0).sum(1)[valid] / cnt[valid].clamp(min=1))
    return loss.mean()


class LossWeighting(nn.Module):
    """Uncertainty weighting of the two objectives.

    L = exp(-s1) * L_ce + s1 + exp(-s2) * L_con + s2, with s1, s2 learned log
    variances. The objective is stationary in s_i at s_i = log L_i, so each
    weight settles at the scale of its own term.
    """

    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(2))

    def forward(self, l_ce, l_con):
        a = torch.exp(-self.s[0])
        b = torch.exp(-self.s[1])
        total = a * l_ce + self.s[0] + b * l_con + self.s[1]
        return total, (a.detach().item(), b.detach().item())


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class Unimodal(nn.Module):
    def __init__(self, which="ts_transformer", positional=True, freeze_vit=False):
        super().__init__()
        self.which = which
        if which == "ts_transformer":
            self.enc = TSTransformer(positional=positional)
        elif which == "ts_lstm":
            self.enc = TSLSTM()
        elif which == "img_vit":
            self.enc = ImgViT(freeze=freeze_vit)
        else:
            raise ValueError(which)
        self.head = nn.Linear(D, 2)
        self.uses_img = which.startswith("img")

    def forward(self, ts, img):
        z = self.enc(img if self.uses_img else ts)
        return dict(logits=self.head(z), z_ts=z, z_img=z, alpha=None)


class EarlyConcat(nn.Module):
    """Raw-feature concatenation: flattened MVTS with a downsampled raw image."""

    def __init__(self, t=60, f=24, img_px=32):
        super().__init__()
        self.img_px = img_px
        dim = t * f + img_px * img_px
        self.net = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(512, D), nn.ReLU())
        self.head = nn.Linear(D, 2)

    def forward(self, ts, img):
        g = img.mean(dim=1, keepdim=True)                      # back to one channel
        g = Fn.adaptive_avg_pool2d(g, self.img_px).flatten(1)
        x = torch.cat([ts.flatten(1), g], dim=1)
        z = self.net(x)
        return dict(logits=self.head(z), z_ts=z, z_img=z, alpha=None)


class LateAvg(nn.Module):
    """Two unimodal branches, each with its own head; logits averaged."""

    def __init__(self, positional=True, freeze_vit=False):
        super().__init__()
        self.ts = TSTransformer(positional=positional)
        self.img = ImgViT(freeze=freeze_vit)
        self.head_ts = nn.Linear(D, 2)
        self.head_img = nn.Linear(D, 2)

    def forward(self, ts, img):
        z_ts, z_img = self.ts(ts), self.img(img)
        lt, li = self.head_ts(z_ts), self.head_img(z_img)
        return dict(logits=0.5 * (lt + li), logits_ts=lt, logits_img=li,
                    z_ts=z_ts, z_img=z_img, alpha=None)


class EmbedConcat(nn.Module):
    """Feature-level fusion: the two embeddings concatenated before the head."""

    def __init__(self, positional=True, freeze_vit=False):
        super().__init__()
        self.ts = TSTransformer(positional=positional)
        self.img = ImgViT(freeze=freeze_vit)
        self.head = nn.Linear(2 * D, 2)

    def forward(self, ts, img):
        z_ts, z_img = self.ts(ts), self.img(img)
        return dict(logits=self.head(torch.cat([z_ts, z_img], dim=1)),
                    z_ts=z_ts, z_img=z_img, alpha=None)


class DualTFusion(nn.Module):
    def __init__(self, ts_enc="transformer", img_enc="vit", fusion="sigmoid",
                 positional=True, freeze_vit=False, gate_init_bias=2.0):
        super().__init__()
        self.ts = {"transformer": lambda: TSTransformer(positional=positional),
                   "lstm": lambda: TSLSTM(),
                   "conv": lambda: TSConv()}[ts_enc]()
        self.img = ImgViT(freeze=freeze_vit) if img_enc == "vit" else ImgConv()
        self.fusion = ScalarGateFusion(mode=fusion, gate_init_bias=gate_init_bias)
        self.head = nn.Linear(D, 2)

    def forward(self, ts, img):
        z_ts, z_img = self.ts(ts), self.img(img)
        z, alpha = self.fusion(z_ts, z_img)
        return dict(logits=self.head(z), z_ts=z_ts, z_img=z_img, alpha=alpha)


# --------------------------------------------------------------------------
# registry: name -> (builder, uses supervised contrastive loss, image source)
# --------------------------------------------------------------------------
def _d(**kw):
    return lambda: DualTFusion(**kw)


REGISTRY = {
    # unimodal baselines
    "ts_transformer":   (lambda: Unimodal("ts_transformer"), False, "crop"),
    "ts_lstm":          (lambda: Unimodal("ts_lstm"), False, "crop"),
    "img_vit":          (lambda: Unimodal("img_vit"), False, "crop"),
    "img_vit_fulldisk": (lambda: Unimodal("img_vit"), False, "full"),
    # alternative encoder architectures
    "dual_conv":        (_d(ts_enc="conv", img_enc="conv", fusion="sigmoid"), True, "crop"),
    "lstm_conv":        (_d(ts_enc="lstm", img_enc="conv", fusion="sigmoid"), True, "crop"),
    # multimodal fusion baselines
    "early_concat":     (lambda: EarlyConcat(), False, "crop"),
    "late_avg":         (lambda: LateAvg(), False, "crop"),
    "embed_concat":     (lambda: EmbedConcat(), False, "crop"),
    # proposed model
    "dualt":            (_d(fusion="sigmoid"), True, "crop"),
    # ablations of the proposed model
    "dualt_fulldisk":   (_d(fusion="sigmoid"), True, "full"),
    "dualt_nopos":      (_d(fusion="sigmoid", positional=False), True, "crop"),
    "dualt_nosupcon":   (_d(fusion="sigmoid"), False, "crop"),
    "dualt_nogate":     (_d(fusion="none"), True, "crop"),
    "dualt_inert":      (_d(fusion="inert"), True, "crop"),
    "dualt_frozenvit":  (_d(fusion="sigmoid", freeze_vit=True), True, "crop"),
}

# configurations with no ViT backbone, trained at the higher head learning rate
VIT_FREE = {"ts_transformer", "ts_lstm", "early_concat", "dual_conv", "lstm_conv"}
