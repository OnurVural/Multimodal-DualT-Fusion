"""Cache active-region-cropped 224x224 AIA images, with crop diagnostics.

Each image is cropped to the active region its paired series describes, using
the heliographic projection in geom.py. Instances whose region coordinates are
missing fall back to the full disk and are counted in the report.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import read_jp2_wcs, ar_bbox_pixels, heliocentric_angle

Image.MAX_IMAGE_PIXELS = None
BASE = os.environ.get("SWANSF_IMG_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "DATA", "SWAN-SF_IMG"))
# argv: [out_dirname] [stretch_mode]   stretch_mode in {"log", "linear"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   sys.argv[1] if len(sys.argv) > 1 else "cache_crop")
STRETCH_MODE = sys.argv[2] if len(sys.argv) > 2 else "log"
# argv[3] == "nocrop": keep the full disk but apply the SAME stretch, so that
# crop-vs-full-disk isolates localization rather than conflating it with the
# contrast normalization.
NO_CROP = (len(sys.argv) > 3 and sys.argv[3] == "nocrop")
SIZE = 224
COORD = ["LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "CRLT_OBS", "HC_ANGLE"]


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


def stretch(a):
    """Percentile clip, then to uint8.

    "log": clip to [p1, p99.5] then log1p. Helioviewer JP2s are already
           byte-scaled for display, so this applies a SECOND log compression
           and crushes ~69% of crop pixels above 200/255.
    "linear": clip to [p0.5, p99.5] and rescale linearly, preserving contrast.
    """
    plo = 1.0 if STRETCH_MODE == "log" else 0.5
    lo, hi = np.percentile(a, plo), np.percentile(a, 99.5)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        lo, hi = float(a.min()), float(max(a.max(), a.min() + 1))
    a = np.clip(a, lo, hi)
    if STRETCH_MODE == "log":
        a = np.log1p(a - lo)
        m = a.max()
        return (a / m * 255.0).astype(np.uint8) if m > 0 else np.zeros_like(a, np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def work(args):
    csv_path, jp2_path = args
    diag = dict(cropped=0, dt_min=float("nan"), hc_err=float("nan"), size_px=0)
    try:
        if not os.path.exists(jp2_path):
            return None, None, "missing jp2", diag
        wcs = read_jp2_wcs(jp2_path)
        df = pd.read_csv(csv_path, sep="\t")
        ts = _naive(df["Timestamp"])
        d = (ts - _naive(wcs["DATE-OBS"])).abs()
        row = df.iloc[int(d.values.argmin())]
        diag["dt_min"] = float(d.dt.total_seconds().min() / 60.0)

        im = Image.open(jp2_path).convert("L")
        arr = np.asarray(im, dtype=np.float32)

        vals = [row.get(c, np.nan) for c in COORD]
        if (not NO_CROP) and np.isfinite(np.array(vals[:5], dtype=float)).all():
            (x0, y0, x1, y1), (cx, cy) = ar_bbox_pixels(row, wcs)
            if x1 - x0 >= 32 and y1 - y0 >= 32:
                arr = arr[y0:y1, x0:x1]
                diag["cropped"] = 1
                diag["size_px"] = int(x1 - x0)
                latc = 0.5 * (float(row["LAT_MIN"]) + float(row["LAT_MAX"]))
                lonc = 0.5 * (float(row["LON_MIN"]) + float(row["LON_MAX"]))
                if np.isfinite(row.get("HC_ANGLE", np.nan)):
                    diag["hc_err"] = float(abs(heliocentric_angle(
                        latc, lonc, row["CRLT_OBS"]) - float(row["HC_ANGLE"])))
        out = np.asarray(Image.fromarray(stretch(arr)).resize((SIZE, SIZE),
                                                              Image.BILINEAR), dtype=np.uint8)
        return out, None, None, diag
    except Exception as e:
        return None, None, "%s: %s" % (type(e).__name__, e), diag


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for p in range(1, 6):
        part = "partition%d" % p
        jobs, order = [], []
        for cls in ["FL", "NF"]:
            d = os.path.join(BASE, part, cls)
            for csv_path in sorted(glob.glob(os.path.join(d, "*.csv"))):
                base = os.path.splitext(os.path.basename(csv_path))[0]
                jobs.append((csv_path, os.path.join(d, base + "_AIA.jp2")))
                order.append(os.path.join(cls, base))

        with Pool(10) as pool:
            res = pool.map(work, jobs, chunksize=8)

        imgs, names, diags, bad = [], [], [], []
        for (img, _, err, diag), nm in zip(res, order):
            if err is not None or img is None:
                bad.append((nm, err)); continue
            imgs.append(img); names.append(nm); diags.append(diag)

        arr = np.stack(imgs).astype(np.uint8)
        np.save(os.path.join(OUT, "%s_img.npy" % part), arr)
        with open(os.path.join(OUT, "%s_names.json" % part), "w") as fh:
            json.dump(names, fh)
        with open(os.path.join(OUT, "%s_diag.json" % part), "w") as fh:
            json.dump(diags, fh)

        dts = np.array([d["dt_min"] for d in diags], dtype=float)
        hce = np.array([d["hc_err"] for d in diags], dtype=float)
        szs = np.array([d["size_px"] for d in diags], dtype=float)
        report[part] = dict(
            n=len(names), cropped=int(sum(d["cropped"] for d in diags)),
            fell_back=int(len(names) - sum(d["cropped"] for d in diags)),
            dropped=len(bad),
            dt_median_min=float(np.nanmedian(dts)), dt_p95_min=float(np.nanpercentile(dts, 95)),
            dt_max_min=float(np.nanmax(dts)),
            hc_err_median=float(np.nanmedian(hce)), hc_err_p95=float(np.nanpercentile(hce, 95)),
            crop_px_median=float(np.nanmedian(szs[szs > 0])) if (szs > 0).any() else 0.0,
            blank=int((arr.reshape(len(arr), -1).max(1) == 0).sum()))
        print(part, report[part], flush=True)
        if bad:
            print("  dropped:", bad[:5], flush=True)

    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
