#!/bin/bash

timestamp=$(date --utc +%FT%T.%4NZ)
EXP_NAME=exp03_$timestamp

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=$DIR/..

cd $ROOT/src

accelerate launch --config_file accelerate_config.yaml train.py --data-path $ROOT/data --results-path $ROOT/results/$EXP_NAME/ --block-out-channels 32 --layers-per-block 1 --norm-num-groups 16 --eval-every 20