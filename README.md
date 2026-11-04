# DualT-Fusion

Code and results for *DualT-Fusion: Multimodal Representation Learning with Dual
Transformers and Supervised Contrastive Alignment for Solar Flare Prediction*.

DualT-Fusion predicts whether an active region will produce an M- or X-class
flare within the next 24 hours from two views of that region: the SWAN-SF
multivariate time series of photospheric magnetic field parameters, and the
corresponding SDO/AIA 335 Å image. A time series transformer and a Vision
Transformer encode the two modalities, a gated residual module merges them in a
shared embedding space, and a supervised contrastive objective aligns the two
embeddings by class.

---

## Contents

```
geom.py                      heliographic -> image-plane projection
build_cache.py               full-disk image and MVTS cache
build_crop_cache.py          active-region-cropped image cache
verify_geom.py               validation of the projection and the crops
dataset_table.py             dataset composition (Table I)

framework.py                 encoders, gated residual fusion, objectives, registry
train.py                     training and evaluation harness
baselines_classical.py       logistic regression, SVM, random forest, majority
run_experiments.sh           reproduces every reported run
make_tables.py               aggregates the run records into the reported tables

make_fig_ar.py               active region localization figure
make_fig_representations.py  t-SNE and attention rollout figures

results/main/                run records, weight decay 0.1
results/wd001/               run records, weight decay 0.01
```

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pillow needs OpenJPEG support to read the AIA JPEG 2000 images:

```bash
python3 -c "from PIL import features; print(features.check('jpg_2000'))"
```

The reported runs were produced on Apple silicon through the PyTorch MPS
backend (`--device mps`). `--device cuda` and `--device cpu` also work.

---

## Data

Two sources are required and are not redistributed here.

**SWAN-SF** provides the multivariate time series: five temporally
non-overlapping segments of tab-separated files, each instance a 60 × 24 window
covering a 12-hour observation window with a 24-hour prediction horizon.
Instances are labeled by the most intense flare in that horizon, consolidated
into flare-positive (M, X) and flare-negative (FQ, B, C).

**SDO/AIA 335 Å** provides the imagery: full-disk JPEG 2000 observations from
the Helioviewer archive, one paired to each instance by timestamp.

The two are expected under `DATA/SWAN-SF_IMG/partitionN/{FL,NF}/`, each instance
a `.csv` with its `_aia.jp2` alongside. Set `SWANSF_IMG_ROOT` to use a different
location.

### Building the caches

```bash
python3 -I build_cache.py                            # full-disk images + MVTS
python3 -I build_crop_cache.py cache_crop linear     # active region crops
python3 -I build_crop_cache.py cache_full linear nocrop
```

Both image caches use the same intensity scaling, so that comparing cropped
against full-disk inputs isolates the crop rather than the contrast
normalization. Each cache is roughly 450 MB and is regenerated from the raw
data, so neither is tracked here.

### Active region localization

Each SWAN-SF record carries the Stonyhurst bounding box of its active region.
Given the observer tilt `B0` and the solar radius `R` in pixels, a point at
latitude φ and longitude λ projects to

```
x = CRPIX1 + R·cos(φ)·sin(λ)
y = CRPIX2 − R·(sin(φ)·cos(B0) − cos(φ)·cos(λ)·sin(B0))
```

with `R`, `CRPIX` and the plate scale read from each image's world coordinate
system metadata. `verify_geom.py` checks the transform against the `HC_ANGLE`
field stored with each record, confirms every projected centre falls on the
disk, and compares the mean 335 Å intensity inside each crop against a
same-sized box at a random on-disk location.

---

## Reproducing the results

```bash
./run_experiments.sh        # every reported run
python3 -I make_tables.py   # the reported tables
```

Each stage is resumable: a `(configuration, segment pair, seed)` whose result
file already exists is skipped, so the script can be interrupted and restarted.

To run one configuration directly:

```bash
python3 -I train.py --configs dualt --seeds 0 1 2 \
    --pairs 1-2 2-3 3-4 4-5 --epochs 6 --wd 0.1 --out results/main
```

### Protocol

Models are trained on one temporal segment and tested on the next, giving the
four consecutive pairs S₁–S₂ through S₄–S₅, each crossed with three random
seeds for twelve runs per configuration. Within each training segment, 20% of
the **active regions** in each class are held out for validation — not 20% of
the instances, since a flare-positive region contributes tens of overlapping
windows and an instance-level split would place near-duplicates of a validation
sample into training. Validation selects the training epoch and the decision
threshold; each test segment is used once, for reporting.

TSS is the primary metric. The constructed dataset is majority-positive, which
inverts the operational base rate, so accuracy, precision and F1 are base-rate
dependent in the wrong direction.

---

## Run records

`results/` holds one JSON per `(configuration, segment pair, seed)`. Each
records the selected epoch, the validation threshold, the full metric set at
both that threshold and at a fixed 0.5, the per-epoch trajectory with the
learned loss weights, the split sizes, the parameter count and the measured
latency. The reported tables are computed from these files by `make_tables.py`.

`results/main/` holds the runs at weight decay 0.1, which is the fixed recipe
for the proposed model and its ablations. `results/wd001/` holds the second
weight-decay value, at which each baseline was also trained; every baseline is
reported at whichever of the two scored higher.

---

## Model configurations

Selected by name with `--configs`, defined in the registry at the bottom of
`framework.py`.

| Name | |
|---|---|
| `dualt` | the proposed model |
| `ts_transformer`, `ts_lstm` | unimodal time series baselines |
| `img_vit` | unimodal ViT on the active region crop |
| `img_vit_fulldisk` | the same ViT on the full disk |
| `dual_conv`, `lstm_conv` | alternative encoder architectures, same fusion |
| `early_concat`, `late_avg`, `embed_concat` | raw, decision-level and feature-level fusion baselines |
| `dualt_fulldisk` | ablation: full-disk input |
| `dualt_nopos` | ablation: no positional encoding |
| `dualt_nosupcon` | ablation: no contrastive term |
| `dualt_nogate`, `dualt_inert` | ablation: ungated residual sum |
| `dualt_frozenvit` | ablation: frozen ViT backbone |

---
