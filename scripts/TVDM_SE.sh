#!/bin/bash

source activate vdm
srun python -u src/train.py --results-path results/exp11 --use-encoder --train-num-steps 100000 --eval-every 1000 --num-workers 3 --batch-size 64 --n-blocks 8 --embedding-dim 32 --block-out-channels 32 --layers-per-block 1 --norm-num-groups 16 --w-dim 128