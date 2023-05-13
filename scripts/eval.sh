#!/bin/bash

EXP_PATH=$(pwd)/$1  # input argument

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=$DIR/..

cd $ROOT/src

accelerate launch --config_file accelerate_config.yaml eval.py --batch-size 32 --data-path $ROOT/data --results-path $EXP_PATH/