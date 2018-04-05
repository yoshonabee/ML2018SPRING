import sys
import numpy as np
import csv
import math

def sign(x): return 1 if x > 0.5 else 0

noz = [0, 10, 78, 79, 80]
data = []

f = open(sys.argv[3], 'r') 
row = csv.reader(f , delimiter=",")
n_row = 0
for r in row:
	if n_row != 0:
		temp = []
		for i in range(len(r)):	temp.append(int(r[i]))
		data.append(temp)
	n_row += 1
f.close()

y = []
f = open(sys.argv[4], 'r') 
ys = f.read()
for i in ys:
	if i == '0':
		y.append(0)
	elif i == '1':
		y.append(1)
f.close()

x = data
lx = len(x)

pc = [0, 0]
pxc = []
for i in y: pc[i] += 1

for i in range(2):
	c = []
	for j in range(len(x[0])):
		temp = [0, 0]
		if j not in noz:
			for xs in range(lx):
				if y[xs] == i: temp[x[xs][j]] += 1
			for m in range(2):
				temp[m] /= pc[i]
			c.append(temp)
		else:
			mean = 0.
			temp = [0, 0]
			for xs in range(lx):
				mean += x[xs][j]
			mean /= lx
			for xs in range(lx):
				if y[xs] == i: temp[x[xs][j] > mean] += 1
			for m in range(2):
				temp[m] /= pc[i]
			c.append(temp)
	pxc.append(c)

for i in range(2):
	pc[i] /= lx;

test_data = []
f = open(sys.argv[5], 'r')
row = csv.reader(f , delimiter=",")
n_row = 0
for r in row:
	if n_row != 0:
		temp = []
		for i in range(len(r)): temp.append(int(r[i]))
		test_data.append(temp)
	n_row += 1
f.close()

xt = test_data
lxt = len(xt)

mean = [0] * len(xt[0])
for feature in noz:
	for xs in range(lxt):
		mean[feature] += xt[xs][feature] / lxt

yt = []
for i in range(lxt):
	up, down = 1, 1
	for feature in range(len(xt[i])):
		if feature not in noz:
			up *= pxc[1][feature][xt[i][feature]]
			down *= pxc[0][feature][xt[i][feature]]
		else:
			up *= pxc[1][feature][xt[i][feature] > mean[feature]]
			down *= pxc[0][feature][xt[i][feature] > mean[feature]]

	up *= pc[1]
	down *= pc[0]
	yt.append(sign(up / (up + down)))



f = open(sys.argv[6], 'w')
lines = ['id,label\n']

for i in range(len(yt)):
    lines.append(str(i + 1) + ',' + str(yt[i]) + '\n')
f.writelines(lines)
f.close()