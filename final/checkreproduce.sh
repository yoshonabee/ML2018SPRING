#!/bin/bash
python ./src/result/ensemble.py $1
python ./src/check.py ./src/en16 $1