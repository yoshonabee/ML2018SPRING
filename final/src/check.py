# -*- coding: UTF-8 -*-
import numpy as np
import csv
import sys
def readtest(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  x = list()
  for i, row in enumerate(rows):
    if i != 0:
      data = row[1].split(' ')
      x.append(list(map(int, data)))  
    # if i == 3:
    #   break

  #x = np.array(x, float)
  # x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
  text.close()
  return np.array(x)

a = readtest(sys.argv[1])
b = readtest(sys.argv[2])
n = 0
for i,j in zip(a,b):
  if i - j != 0:
    n += 1
print('different row:',n)