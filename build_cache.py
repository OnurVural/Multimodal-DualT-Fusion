"""Cache MVTS tensors and full-disk 224x224 images.

  MVTS : 24 magnetic field parameters, per-parameter mean imputation within the
         window (all-NaN parameter -> 0), then per-instance z-score (std 0 -> 1)
  Image: grayscale, bilinear resize to 224x224. Channel replication and
         ImageNet normalization are applied at training time.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from multiprocessing import Pool

BASE = os.environ.get("SWANSF_IMG_ROOT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "DATA", "SWAN-SF_IMG"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
RESIZE = transforms.Resize((224, 224))
T, F = 60, 24


def load_ts(csv_path):
    df = pd.read_csv(csv_path, sep="\t").iloc[:, 1:25]
    df = df.astype(float)
    for col in df.columns:
        if df[col].isna().all():
            df[col] = 0.0
        elif df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    std[std == 0] = 1.0
    return ((df - mean) / std).to_numpy(dtype=np.float32)


def load_img(aia_path):
    if os.path.exists(aia_path):
        img = Image.open(aia_path).convert("L")
        missing = 0
    else:
        img = Image.new("L", (128, 128), color=0)
        missing = 1
    return np.asarray(RESIZE(img), dtype=np.uint8), missing


def work(args):
    csv_path, aia_path = args
    try:
        ts = load_ts(csv_path)
        if ts.shape != (T, F):
            return (csv_path, None, None, 0, "shape=%s" % (ts.shape,))
        img, missing = load_img(aia_path)
        return (csv_path, ts, img, missing, None)
    except Exception as e:
        return (csv_path, None, None, 0, "%s: %s" % (type(e).__name__, e))


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for p in range(1, 6):
        part = "partition%d" % p
        jobs, labels, names = [], [], []
        for label_name in ["FL", "NF"]:
            d = os.path.join(BASE, part, label_name)
            label = 1 if label_name == "FL" else 0
            for csv_path in sorted(glob.glob(os.path.join(d, "*.csv"))):
                base = os.path.splitext(os.path.basename(csv_path))[0]
                jobs.append((csv_path, os.path.join(d, base + "_AIA.jp2")))
                labels.append(label)
                names.append(os.path.join(label_name, base))

        with Pool(10) as pool:
            res = pool.map(work, jobs, chunksize=8)

        ok_ts, ok_img, ok_lab, ok_name = [], [], [], []
        bad, missing = [], 0
        for (csv_path, ts, img, miss, err), lab, nm in zip(res, labels, names):
            if err is not None:
                bad.append((nm, err))
                continue
            ok_ts.append(ts); ok_img.append(img); ok_lab.append(lab); ok_name.append(nm)
            missing += miss

        ts_arr = np.stack(ok_ts).astype(np.float32)
        img_arr = np.stack(ok_img).astype(np.uint8)
        lab_arr = np.asarray(ok_lab, dtype=np.int64)
        np.save(os.path.join(OUT, "%s_ts.npy" % part), ts_arr)
        np.save(os.path.join(OUT, "%s_img.npy" % part), img_arr)
        np.save(os.path.join(OUT, "%s_y.npy" % part), lab_arr)
        with open(os.path.join(OUT, "%s_names.json" % part), "w") as fh:
            json.dump(ok_name, fh)

        nan_ts = int(np.isnan(ts_arr).sum())
        report[part] = dict(n=len(ok_lab), pos=int(lab_arr.sum()), neg=int((lab_arr == 0).sum()),
                            missing_img=missing, dropped=len(bad), nan_in_ts=nan_ts,
                            blank_img=int((img_arr.reshape(len(img_arr), -1).max(1) == 0).sum()))
        print(part, report[part], flush=True)
        if bad:
            print("  dropped examples:", bad[:5], flush=True)

    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
