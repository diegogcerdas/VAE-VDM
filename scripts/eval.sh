#!/bin/bash

EXP_NAME=$1  # input argumennt

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=$DIR/..

cd $ROOT/src

accelerate launch --config_file accelerate_config.yaml eval.py --batch-size 32 --data-path $ROOT/data --results-path $ROOT/results/$EXP_NAME/