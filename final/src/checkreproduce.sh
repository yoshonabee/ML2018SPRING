#!/bin/bash
python ./result/ensemble.py reproduce.csv
python check.py en16 reproduce.csv