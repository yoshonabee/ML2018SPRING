import os
import sys
import torch
import numpy as np
from hw5_fc import *
from torch.optim import Adam
from gensim.models import Word2Vec
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

BATCH_SIZE = 256
SENTENCE_LENGTH = 40

wordVec = Word2Vec.load('wv.h5')
wordVec = wordVec.wv

x_test = loadData(sys.argv[1], sentence_length=SENTENCE_LENGTH, train=False)
model = torch.load('model.pt')

test_data = Data(x_test, y=None, word_vector=wordVec)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE * 2, shuffle=False)

print('----------------------Predicting------------------------')
y_test = torch.LongTensor()
for i, x in enumerate(test_loader):
	x = Variable(x.cuda())
	
	output = model(x)

	pred = output.cpu().data.max(1, keepdim=True)[1].long()
	y_test = torch.cat((y_test, pred), dim=0)
	
		
	if (i + 1) % 10 == 0: print('Iteration: %d' %(i + 1))

y_test = y_test.numpy()[:,0]
print(y_test.shape)
outputcsv(y_test, sys.argv[2])