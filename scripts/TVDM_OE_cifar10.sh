#!/bin/bash

source activate vdm
srun python -u src/train.py --results-path results/exp10_cifar10 --no-use-mnist --use-encoder --train-num-steps 100000 --eval-every 1000 --num-workers 3 --batch-size 64 --n-blocks 8 --embedding-dim 32 --block-out-channels 64 --layers-per-block 2 --norm-num-groups 32 --w-dim 128
