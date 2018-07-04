# -*- coding: UTF-8 -*-
import numpy as np
import csv
import pandas as pd
import sys


def readtest(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  x = list()
  for i, row in enumerate(rows):
    if i != 0:
      data = row[1:]
      x.append(list(map(float, data)))  
  text.close()
  return np.array(x)

a = readtest(sys.argv[1])
b = readtest(sys.argv[2])
print(np.sum(a-b,axis=0))