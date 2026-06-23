#!/bin/bash
set -ex
/home/sankhya/Python/bin/python3 train.py \
  --dataroot ./datasets/guwahati_azara_processed \
  --name guwahati_azara_pix2pix \
  --model pix2pix \
  --direction AtoB \
  --netG unet_256 \
  --dataset_mode aligned \
  --norm batch \
  --pool_size 0 \
  --batch_size 4 \
  --n_epochs 25 \
  --n_epochs_decay 25

