"""Figure: full-disk AIA input vs the extracted active-region crop, for one
flare-positive and one flare-negative instance.

Left column  : full disk with the projected active-region bounding box
               overlaid.
Right column : the AR crop actually fed to the vision encoder.
"""
import glob, os, sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geom import read_jp2_wcs, ar_bbox_pixels

Image.MAX_IMAGE_PIXELS = None
BASE = os.environ.get("SWANSF_IMG_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "DATA", "SWAN-SF_IMG"))
DISPLAY = 768          # downsample the full disk for display/file size
OUT_PDF = os.path.join(HERE, "fig_ar_localization.pdf")
OUT_PNG = os.path.join(HERE, "fig_ar_localization.png")


def _naive(x):
    x = pd.to_datetime(x)
    try:
        if getattr(x, "tz", None) is not None:
            x = x.tz_convert(None)
    except (TypeError, AttributeError):
        pass
    try:
        if getattr(x.dt, "tz", None) is not None:
            x = x.dt.tz_convert(None)
    except (TypeError, AttributeError):
        pass
    return x


def stretch(a, plo=0.5, phi=99.5):
    """Linear percentile contrast stretch.

    Helioviewer JP2s are already byte-scaled for display, so an additional log
    compresses them into the top of the range (69% of crop pixels above 200/255);
    a plain percentile stretch keeps the structure legible.
    """
    lo, hi = np.percentile(a, plo), np.percentile(a, phi)
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max() + 1)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def pick(cls, want_label):
    """Choose a well-posed example: close in time, near disk centre, decent size."""
    d = os.path.join(BASE, "partition1", cls)
    best = None
    for csv_path in sorted(glob.glob(os.path.join(d, "*.csv")))[:400]:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        jp2 = os.path.join(d, base + "_AIA.jp2")
        if not os.path.exists(jp2):
            continue
        try:
            wcs = read_jp2_wcs(jp2)
            df = pd.read_csv(csv_path, sep="\t")
            dt = (_naive(df["Timestamp"]) - _naive(wcs["DATE-OBS"])).abs()
            if dt.dt.total_seconds().min() > 1800:
                continue
            row = df.iloc[int(dt.values.argmin())]
            vals = [row["LAT_MIN"], row["LAT_MAX"], row["LON_MIN"], row["LON_MAX"],
                    row["CRLT_OBS"], row["HC_ANGLE"]]
            if not np.isfinite(np.array(vals, dtype=float)).all():
                continue
            if float(row["HC_ANGLE"]) > 45:          # avoid limb foreshortening
                continue
            (x0, y0, x1, y1), _ = ar_bbox_pixels(row, wcs)
            size = x1 - x0
            if not (300 <= size <= 900):
                continue
            score = abs(size - 550) + float(row["HC_ANGLE"]) * 3
            if best is None or score < best[0]:
                best = (score, jp2, (x0, y0, x1, y1), base, want_label)
        except Exception:
            continue
    return best


def main():
    picks = [pick("FL", "Flare-positive"), pick("NF", "Flare-negative")]
    if any(p is None for p in picks):
        raise SystemExit("could not find suitable examples")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.1))
    for r, (_, jp2, box, base, label) in enumerate(picks):
        x0, y0, x1, y1 = box
        im = Image.open(jp2).convert("L")
        full = np.asarray(im, dtype=np.float32)
        crop = full[y0:y1, x0:x1]

        shown = np.asarray(Image.fromarray(stretch(full) * 255).convert("L")
                           .resize((DISPLAY, DISPLAY), Image.BILINEAR), dtype=np.float32)
        sc = DISPLAY / full.shape[0]

        ax = axes[r, 0]
        ax.imshow(shown, cmap="gray", origin="upper")
        ax.add_patch(Rectangle((x0 * sc, y0 * sc), (x1 - x0) * sc, (y1 - y0) * sc,
                               fill=False, lw=1.6, edgecolor="#e34a33"))
        ax.set_title("Full disk ($4096^2$), AR outlined", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[r, 1]
        ax.imshow(np.asarray(Image.fromarray(stretch(crop) * 255).convert("L")
                             .resize((224, 224), Image.BILINEAR)),
                  cmap="gray", origin="upper")
        ax.set_title("Extracted AR crop ($224^2$ input)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#e34a33"); s.set_linewidth(1.6)
        ar = base.split("_ar")[1].split("_")[0] if "_ar" in base else "?"
        ax.set_xlabel("NOAA/HARP region %s" % ar, fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    for p in (OUT_PDF, OUT_PNG):
        print("wrote %s (%.0f KB)" % (p, os.path.getsize(p) / 1024))


if __name__ == "__main__":
    main()
