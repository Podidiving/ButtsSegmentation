#!/usr/bin/env bash


# train
python3 prepare_masks.py -p data/cig_butts/train -o data/cig_butts/train/masks

# val
python3 prepare_masks.py -p data/cig_butts/val -o data/cig_butts/val/masks
