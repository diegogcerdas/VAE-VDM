#!/bin/bash

source activate vdm
srun python -u src/train.py --results-path results/exp12 --use-encoder --train-num-steps 100000 --eval-every 1000 --num-workers 3 --batch-size 64 --n-blocks 8 --embedding-dim 32 --block-out-channels 16 --layers-per-block 1 --norm-num-groups 8 --w-dim 128