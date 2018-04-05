import sys
import numpy as np
import hw2_fc as fc

x = fc.readX(sys.argv[1])
y = fc.readY(sys.argv[2])
lx = len(x)

pc = [0, 0]
pxc = []
for i in y: pc[i] += 1

for i in range(2):
	c = []
	for j in range(len(x[0])):
		temp = [0, 0]
		if j not in fc.noz:
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

np.save("generative_pc.npy", pc)
np.save("generative_pxc.npy", pxc)