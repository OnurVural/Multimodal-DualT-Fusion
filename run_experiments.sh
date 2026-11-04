#!/bin/zsh
# Reproduces every run reported in the paper. Each stage is resumable: train.py
# and baselines_classical.py skip any (config, pair, seed) whose result file
# already exists.
cd "$(dirname "$0")" || exit 1
export TOKENIZERS_PARALLELISM=false
export CROP_CACHE=cache_crop
export FULL_CACHE=cache_full

EP="--epochs 6"
WD10="--wd 0.1"
WD01="--wd 0.01"

PROPOSED="dualt"
CHEAP="ts_transformer ts_lstm early_concat dual_conv lstm_conv"
VIT_BASELINES="img_vit late_avg embed_concat"
ABLATIONS="dualt_fulldisk dualt_nopos dualt_nosupcon dualt_nogate dualt_inert dualt_frozenvit img_vit_fulldisk"

say() { echo "" ; echo "########## $1" ; echo "" ; }

say "1/5  classical and trivial baselines, 3 seeds"
python3 -I baselines_classical.py --seeds 0 1 2 --out results/main

say "2/5  proposed model, 4 pairs x 3 seeds"
python3 -I train.py ${=EP} ${=WD10} --configs ${=PROPOSED} --seeds 0 1 2 --out results/main

say "3/5  deep baselines without a ViT backbone, both weight-decay values"
python3 -I train.py ${=EP} ${=WD10} --configs ${=CHEAP} --seeds 0 1 2 --out results/main
python3 -I train.py ${=EP} ${=WD01} --configs ${=CHEAP} --seeds 0 1 2 --out results/wd001

say "4/5  ViT-based baselines"
python3 -I train.py ${=EP} ${=WD01} --configs ${=VIT_BASELINES} --seeds 0 1 2 --out results/wd001
python3 -I train.py ${=EP} ${=WD10} --configs late_avg embed_concat --seeds 0 --pairs 1-2 --out results/main

say "5/5  ablations of the proposed model, seed 0"
python3 -I train.py ${=EP} ${=WD10} --configs ${=ABLATIONS} --seeds 0 --out results/main

echo ""
echo "done. aggregate with: python3 -I make_tables.py"
