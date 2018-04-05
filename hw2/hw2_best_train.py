import sys
import numpy as np
import hw2_fc as fc

x = fc.readX(sys.argv[1])
y = fc.readY(sys.argv[2])

xval = x[len(x) * 4 // 5:len(x)]
x = x[0:len(x) * 4 // 5]
yval = y[len(y) * 4 // 5:len(y)]
y = y[0:len(y) * 4 // 5]
lx = len(x)
lxval = len(xval)

x = fc.fit(x)
xval = fc.fit(xval)

x = np.array(x)
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
y = np.array(y)
yval = np.array(yval)
xval = np.array(xval)
xval = np.concatenate((np.ones((xval.shape[0],1)),xval), axis=1)

lr = 0.1
w = np.array([1.0] * len(x[0]))
s_gra = np.zeros(len(x[0]))

for iter in range(60):
	for a in range(lx):
		fx = fc.theta(np.dot(w, x[a]))
		gra = (fx - y[a]) * x[a]
		s_gra += gra ** 2
		ada = np.sqrt(s_gra)
		w -= lr * gra / ada

	err = 0
	for i in range(lx):
		if fc.sign(fc.theta(np.dot(w, x[i]))) != y[i]: err += 1

	errval = 0
	for i in range(lxval):
		if fc.sign(fc.theta(np.dot(w, xval[i]))) != yval[i]: errval += 1
	print("Iteration = %d | Accuracyin = %f | Accruacyout = %f"%(iter, 1 - err / lx, 1 - errval / lxval))

np.save("bestw.npy", w)