#!/bin/sh

set -eu

# How to use
python runner.py \
       --model=xgboost_cfg.py \
       --model.hparams.reg_alpha=0.1
