import sys
import numpy as np
import numpy.linalg as LA
from hw4_pca_fc import *

x = loadImages(sys.argv[1])

x_mean = meanImages(x)

print("\n--------------------SVD--------------------")
U, sigma, v = LA.svd(np.transpose(x - x_mean), full_matrices=False)

print("\n---------------Reconstructing--------------")
idx = idxImage(sys.argv[2])
K = np.dot(x[idx] - x_mean, U[:,0:4])
reconstruct = np.dot(U[:,0:4], K.T)
saveImage('reconstruction.jpg', reconstruct + x_mean)