#!/bin/sh

set -eu

# How to use
python runner.py \
       --model=model.py:ridge \
       --model.hparams.alpha=0.1
