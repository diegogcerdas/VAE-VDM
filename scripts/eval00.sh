#!/bin/bash

EXP_NAME=exp00_test_2023-05-13T12:38:21.4321Z

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=$DIR/..

cd $ROOT/src

accelerate launch --config_file accelerate_config.yaml eval.py --data-path $ROOT/data --results-path $ROOT/results/$EXP_NAME/ --batch-size 32