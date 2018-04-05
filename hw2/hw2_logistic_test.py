import sys
import numpy as np
import hw2_fc as fc

w = np.load("logisticw.npy")

xt = fc.readX(sys.argv[1])
xt = fc.fit(xt)
xt = np.array(xt)
xt = np.concatenate((np.ones((xt.shape[0],1)),xt), axis=1)

yt = []
for i in range(len(xt)):
	yt.append(fc.sign(fc.theta(np.dot(w, xt[i]))))

fc.outputcsv(yt, sys.argv[2])