# -*- coding: UTF-8 -*-
import numpy as np
import csv
import pandas as pd
import sys

file_data = [
  # '0.46600.raw.csv', 'o0.4632.raw.csv', '0.47mabe.raw.csv', 
  # 'other00.raw.csv', 'other01.raw.csv', 'other02.raw.csv', 'other03.raw.csv', 'other04.raw.csv',
  # 'ver0_0.491.raw.csv', 'ver1_0.49.raw.csv', 'ver2_0.49.raw.csv', 'ver5_0.50.raw.csv', 'ver6_0.50.raw.csv', 
  # '150over3.raw.csv', '150over4.raw.csv', 
  # '1253s.raw.csv', '1254s.raw.csv', 
  # 's0.46719.raw.csv', 's0.47312.raw.csv', 's0.47549.raw.csv', 's0.47905.raw.csv', 
  # '0.46600.raw.csv', 'o0.4632.raw.csv', '0.47mabe.raw.csv', 
  # 'other00.raw.csv', 'other01.raw.csv', 'other02.raw.csv', 'other03.raw.csv', 'other04.raw.csv',
  # 'ver0_0.491.raw.csv', 'ver1_0.49.raw.csv', 'ver2_0.49.raw.csv', 'ver5_0.50.raw.csv', 'ver6_0.50.raw.csv', 
  # '150over3.raw.csv', '150over4.raw.csv', 
  # '1250s.raw.csv','1253s.raw.csv',
  # 's0.46719.raw.csv', 's0.47312.raw.csv', 's0.47549.raw.csv', 's0.47905.raw.csv', 
  'opca'
]


def readtest(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  x = list()
  for i, row in enumerate(rows):
    data = row
    x.append(list(map(float, data)))  
  text.close()
  return np.array(x)


data = []
for filename in file_data:
  data.append(readtest('./'+filename + '.csv'))

result = data[0]*0

for d in data:
  result = result + d
print(len(result))
print(result)



sample_submit = pd.read_csv("./submission_example.csv")
print(np.array(result).shape)
pred_y = np.argmax(np.array(result).reshape(-1,6),axis=1)
sample_submit["ans"] = pred_y
sample_submit.to_csv(sys.argv[1],index=None)