"""Validate the heliographic->pixel transform before trusting any crop.

Three independent checks per sample:
  1. our heliocentric angle vs the file's own HC_ANGLE column (convention check)
  2. projected AR centre lies inside the solar disk
  3. mean 335A intensity inside the crop vs a same-size box at a random on-disk
     location (active regions are bright at 335A, so the crop should win)
"""
import glob, os, random, sys
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import read_jp2_wcs, ar_bbox_pixels, heliocentric_angle

Image.MAX_IMAGE_PIXELS = None
BASE = os.environ.get("SWANSF_IMG_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "DATA", "SWAN-SF_IMG"))
random.seed(0)


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


def nearest_row(df, date_obs):
    ts = _naive(df["Timestamp"])
    t = _naive(date_obs)
    d = (ts - t).abs()
    return df.iloc[int(d.values.argmin())], int(d.dt.total_seconds().min())


def main(n_per=25):
    hc_err, inside, bright_ratio, dt_min, sizes = [], [], [], [], []
    for part in ["partition1", "partition3", "partition5"]:
        for cls in ["FL", "NF"]:
            d = os.path.join(BASE, part, cls)
            csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
            for csv_path in random.sample(csvs, min(n_per, len(csvs))):
                base = os.path.splitext(os.path.basename(csv_path))[0]
                jp2 = os.path.join(d, base + "_AIA.jp2")
                if not os.path.exists(jp2):
                    continue
                wcs = read_jp2_wcs(jp2)
                df = pd.read_csv(csv_path, sep="\t")
                row, dts = nearest_row(df, wcs["DATE-OBS"])
                dt_min.append(dts / 60.0)
                if not np.isfinite([row["LAT_MIN"], row["LAT_MAX"],
                                    row["LON_MIN"], row["LON_MAX"],
                                    row["CRLT_OBS"]]).all():
                    continue

                latc = 0.5 * (row["LAT_MIN"] + row["LAT_MAX"])
                lonc = 0.5 * (row["LON_MIN"] + row["LON_MAX"])
                ours = heliocentric_angle(latc, lonc, row["CRLT_OBS"])
                hc_err.append(abs(float(ours) - float(row["HC_ANGLE"])))

                (x0, y0, x1, y1), (cx, cy) = ar_bbox_pixels(row, wcs)
                rad = np.hypot(cx - wcs["CRPIX1"], cy - wcs["CRPIX2"])
                inside.append(rad < wcs["R_SUN"])
                sizes.append(x1 - x0)

                im = Image.open(jp2).convert("L")
                a = np.asarray(im, dtype=np.float32)
                crop = a[y0:y1, x0:x1]
                s = x1 - x0
                for _ in range(12):
                    ang = random.uniform(0, 2 * np.pi)
                    rr = random.uniform(0, 0.75) * wcs["R_SUN"]
                    rx = int(wcs["CRPIX1"] + rr * np.cos(ang))
                    ry = int(wcs["CRPIX2"] + rr * np.sin(ang))
                    rx = min(max(rx - s // 2, 0), a.shape[1] - s)
                    ry = min(max(ry - s // 2, 0), a.shape[0] - s)
                    ref = a[ry:ry + s, rx:rx + s]
                    if ref.size:
                        break
                if crop.size and ref.size and ref.mean() > 0:
                    bright_ratio.append(float(crop.mean() / ref.mean()))

    hc_err = np.array(hc_err); br = np.array(bright_ratio)
    print("samples checked            : %d" % len(hc_err))
    print("HC_ANGLE abs error (deg)   : mean=%.3f  median=%.3f  p95=%.3f  max=%.3f"
          % (hc_err.mean(), np.median(hc_err), np.percentile(hc_err, 95), hc_err.max()))
    print("AR centre inside disk      : %d/%d" % (sum(inside), len(inside)))
    print("image-vs-MVTS time offset  : median=%.1f min  p95=%.1f min  max=%.1f min"
          % (np.median(dt_min), np.percentile(dt_min, 95), np.max(dt_min)))
    print("crop size (px)             : median=%d  min=%d  max=%d"
          % (np.median(sizes), np.min(sizes), np.max(sizes)))
    print("brightness crop/random     : median=%.2fx  frac>1: %.2f  (ARs are bright at 335A)"
          % (np.median(br), float((br > 1).mean())))


if __name__ == "__main__":
    main()
