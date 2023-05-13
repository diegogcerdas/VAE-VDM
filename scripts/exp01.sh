#!/bin/bash

timestamp=$(date --utc +%FT%T.%4NZ)
EXP_NAME=exp01_$timestamp

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=$DIR/..

cd $ROOT/src

accelerate launch --config_file accelerate_config.yaml train.py --data-path $ROOT/data --results-path $ROOT/results/$EXP_NAME/ --no-use-encoder