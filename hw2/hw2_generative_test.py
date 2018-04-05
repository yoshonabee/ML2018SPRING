import sys
import numpy as np
import hw2_fc as fc

pc = np.load("generative_pc.npy")
pxc = np.load("generative_pxc.npy")

xt = fc.readX(sys.argv[1])
lxt = len(xt)

mean = [0] * len(xt[0])
for feature in fc.noz:
	for xs in range(lxt):
		mean[feature] += xt[xs][feature] / lxt

yt = []
for i in range(lxt):
	up, down = 1, 1
	for feature in range(len(xt[i])):
		if feature not in fc.noz:
			up *= pxc[1][feature][xt[i][feature]]
			down *= pxc[0][feature][xt[i][feature]]
		else:
			if xt[i][feature] > mean[feature]:
				up *= pxc[1][feature][1]
				down *= pxc[0][feature][1]
			else:
				up *= pxc[1][feature][0]
				down *= pxc[0][feature][0]

	up *= pc[1]
	down *= pc[0]
	yt.append(fc.sign(up / (up + down)))

fc.outputcsv(yt, sys.argv[2])