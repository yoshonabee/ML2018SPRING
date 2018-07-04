#!/bin/bash
python3 ./src/result/ensemble.py $1
python3 ./src/check.py ./src/en16 $1