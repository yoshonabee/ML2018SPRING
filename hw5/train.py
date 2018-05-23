import os
import sys
import time
import torch
import numpy as np
from hw5_fc import *
from torch.optim import Adam
from gensim.models import Word2Vec
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

LR = 0.001
EPOCH = 15
VEC_SIZE = 200
BATCH_SIZE = 256
SENTENCE_LENGTH = 40

print('----------------------Loading Data------------------------')
x_label, y = loadData(sys.argv[1], sentence_length=SENTENCE_LENGTH)
x_nolabel = loadData(sys.argv[2], sentence_length=SENTENCE_LENGTH, label=False)

print('---------------------Word Embedding-----------------------')
wordVec = Word2Vec(x_label + x_nolabel, size=VEC_SIZE, window=5, min_count=5, workers=4)
del x_nolabel

wordVec.save('wv.h5')
wordVec = Word2Vec.load('wv.h5')
wordVec = wordVec.wv

data_label = Data(x_label, y, wordVec)
train_loader_label = DataLoader(data_label, batch_size=BATCH_SIZE, shuffle=True)

print('---------------------Building Model-----------------------')

#Model1

model01 = RNN01()
model01.cuda()
print(model01)
optim = Adam(model01.parameters(), lr=LR, weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()
loss_function.cuda()

for epoch in range(EPOCH):
	iter_start = time.time()
	total_loss = 0
	total_acc = 0
	for iter, (x, y) in enumerate(train_loader_label):
		x, y = Variable(x.cuda()), Variable(y.cuda())

		output = model01(x)
		loss = loss_function(output, y)
		optim.zero_grad()
		loss.backward()
		optim.step()

		total_loss += loss.data[0]
		if (iter + 1) % 20 == 0:
			iter_end = time.time()
			pred = output.cpu().data.max(1, keepdim=True)[1].long().numpy()[:,0]
			train_acc = 0
			for i in range(len(pred)): train_acc += (pred[i] == y.data[i])
			train_acc /= len(pred)
			total_acc += train_acc
			print('Epoch:', epoch + 1, '| Iter:', iter + 1, '\t| train loss:%.4f | train acc:%.4f | time:%.4f'
				%(total_loss / (iter + 1), total_acc * 20/ (iter + 1), iter_end - iter_start))
			iter_start = iter_end

torch.save(model01, 'model01.pt')

#Model2

model02 = RNN02()
model02.cuda()
print(model02)
optim = Adam(model02.parameters(), lr=LR, weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()
loss_function.cuda()

for epoch in range(EPOCH):
	iter_start = time.time()
	total_loss = 0
	total_acc = 0
	for iter, (x, y) in enumerate(train_loader_label):
		x, y = Variable(x.cuda()), Variable(y.cuda())

		output = model02(x)
		loss = loss_function(output, y)
		optim.zero_grad()
		loss.backward()
		optim.step()

		total_loss += loss.data[0]
		if (iter + 1) % 20 == 0:
			iter_end = time.time()
			pred = output.cpu().data.max(1, keepdim=True)[1].long().numpy()[:,0]
			train_acc = 0
			for i in range(len(pred)): train_acc += (pred[i] == y.data[i])
			train_acc /= len(pred)
			total_acc += train_acc
			print('Epoch:', epoch + 1, '| Iter:', iter + 1, '\t| train loss:%.4f | train acc:%.4f | time:%.4f'
				%(total_loss / (iter + 1), total_acc * 20/ (iter + 1), iter_end - iter_start))
			iter_start = iter_end

torch.save(model02, 'model02.pt')

#Model3

model03 = RNN03()
model03.cuda()
print(model03)
optim = Adam(model03.parameters(), lr=LR, weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()
loss_function.cuda()

for epoch in range(EPOCH):
	iter_start = time.time()
	total_loss = 0
	total_acc = 0
	for iter, (x, y) in enumerate(train_loader_label):
		x, y = Variable(x.cuda()), Variable(y.cuda())

		output = model03(x)
		loss = loss_function(output, y)
		optim.zero_grad()
		loss.backward()
		optim.step()

		total_loss += loss.data[0]
		if (iter + 1) % 20 == 0:
			iter_end = time.time()
			pred = output.cpu().data.max(1, keepdim=True)[1].long().numpy()[:,0]
			train_acc = 0
			for i in range(len(pred)): train_acc += (pred[i] == y.data[i])
			train_acc /= len(pred)
			total_acc += train_acc
			print('Epoch:', epoch + 1, '| Iter:', iter + 1, '\t| train loss:%.4f | train acc:%.4f | time:%.4f'
				%(total_loss / (iter + 1), total_acc * 20/ (iter + 1), iter_end - iter_start))
			iter_start = iter_end

torch.save(model03, 'model03.pt')

model = Ensemble(model01, model02, model03)
torch.save(model, 'model.pt')