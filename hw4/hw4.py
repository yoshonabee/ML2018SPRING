import sys
import numpy as np
from hw4_fc import *
from sklearn.cluster import KMeans
from sklearn.decomposition import *

images = np.load(sys.argv[1])

pca = PCA(n_components=300, whiten=True)
x_train = pca.fit_transform(images)
print(x_train.shape)
np.save('x_train.npy', x_train)
del pca, images

model = KMeans(n_clusters=2)
model.fit(x_train)

test_case = loadData(sys.argv[2])

output = []
for i in range(test_case.shape[0]):
	if (i + 1) % 50000 == 0: print(i + 1)
	if (model.labels_[test_case[i][0]] == model.labels_[test_case[i][1]]): output.append(1)
	else: output.append(0)

outputcsv(output, sys.argv[3])