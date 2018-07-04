import numpy as np
import keras.backend as K
import keras
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, GRU, Embedding, Bidirectional, BatchNormalization, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping
from keras.layers.merge import add, dot, concatenate
import numpy as np
import random
from gensim.models import Word2Vec
import pandas as pd
import os, sys
import pickle
# ------------------ uilt ----------
from 300sutils import *
# ----------------------------------

output_path = './oo.csv'
chsplit = False
# --------------train data load
rawX = []
y = []
split = []
for i in range(1,6): # ----from 1~5 file
  with open('./data/training_data/'+str(i)+'_train.txt','r') as trainfile:
    for row in trainfile:
      row = row.replace('\n','').lower()
      ch = splitall(row, chsplit)
      # print(splitChinese(row),len(ch))
      rawX.append(ch)
      
      y.append(i)
  split.append(len(rawX)-1)


# --------------data preprocess --------------------------
vec_dim = 300
Xlist = [ word.split(" ") for word in rawX]
if os.path.exists('./other/w2v_300s'):
  w2v = Word2Vec.load('./other/w2v_300s')
else:
  w2v = embedding(Xlist + [sent.split(' ') for sent in test_tokens], vec_dim)# FastText(Xlist, size=300, window=5, min_count=5, workers=12, sg=1, seed=0)
  w2v.save('./other/w2v_300s')
#   print(w2v.wv['iphone'])
# exit()
print('got',len(Xlist),'data.')    
#------------------token----------------
tknzr = Tokenizer(filters='\t')
# print(rawX[0])
# exit()
tknzr.fit_on_texts(rawX + test_tokens)
with open('./other/tknzr_300s.pickle', 'wb') as tknfile:
    pickle.dump(tknzr, tknfile)
word_seq = tknzr.texts_to_sequences(rawX)
word_idx = tknzr.word_index
#----------------------------------------------
print('rawX[0]:', rawX[0])
print('word_seq[0]:', word_seq[0])
print('word_idx["沒"]:', word_idx["沒"])
wordNum = len(word_idx)
print('got tokens:', wordNum)
embedding_weights = idx2weight(word_idx, vec_dim, w2v)
longestSent = 0
for i in word_seq:
  if len(i) > longestSent:
    longestSent = len(i)

print("longest sentence:", longestSent)

X = pad_sequences(word_seq, maxlen=longestSent)
y = np.array(X[1:].copy())
X = np.array(X[:len(X)-1])
np.delete(X, split, axis=0)
np.delete(y, split, axis=0)

'''
ex: 
longestSentence = 8
[0,0,0,0,10,218,100,12]
numbers n(int) in list is the sequence in embedding_weights[n(int)]
'''
# -----------------------------------------------
# model
batch_size = 1024
epochs = 100


# ======= define model ======
preInput = Input(shape=(longestSent,), dtype='int32')
em1 = Embedding(len(tknzr.word_index) +1,
                output_dim = vec_dim,
                weights=[embedding_weights],
                input_length=longestSent,
                trainable=False)(preInput)
preSent = Bidirectional(GRU(128,dropout=0.2, return_sequences=True))(em1)
preSent = Bidirectional(GRU(64,dropout=0.2))(preSent)
preSent = BatchNormalization()(preSent)
preSent = Dense(256,activation="relu")(preSent)
preSent = Dense(256,activation="relu")(preSent)

posInput = Input(shape=(longestSent,), dtype='int32')
em2 = Embedding(len(tknzr.word_index) +1,
                output_dim= vec_dim, 
                weights=[embedding_weights],
                input_length=longestSent,
                trainable=False)(posInput)
posSent = Bidirectional(GRU(128,dropout=0.2, return_sequences = True))(em2)
posSent = Bidirectional(GRU(64,dropout=0.2))(posSent)
posSent = BatchNormalization()(posSent)
posSent = Dense(256,activation="relu")(posSent)
posSent = Dense(256,activation="relu")(posSent)

merge = keras.layers.dot([preSent, posSent],1)
output_dense = Dense(1,activation="sigmoid")(merge)
model = Model(inputs=[preInput, posInput], outputs=output_dense)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
print(model.summary())

# print(train_data_pad)
for i in range(0,20):
    # ======= train valid split =======
    train_pre, valid_pre, train_pos, valid_pos = train_test_split(X, y, test_size=0.05)
    for epoch in range(epochs):
        # training
        false_pre = []
        false_pos = train_pos.copy()

        true_pre = train_pre
        true_pos = train_pos

        roll_sample = np.random.choice(len(train_pre),1, replace=False)

        # false_caption = np.concatenate((np.roll(train_caption_pad,roll_sample[0],axis=0),
        #                                 # np.roll(train_caption_pad,roll_sample[1],axis=0),
        #                                 # np.roll(train_caption_pad,roll_sample[2],axis=0),
        #                                 # np.roll(train_caption_pad,roll_sample[3],axis=0),
        #                                 # np.roll(train_caption_pad,roll_sample[4],axis=0),))
        #                                 ))
        false_pre = np.concatenate((np.roll(train_pre,roll_sample[0],axis=0),
                                        # np.roll(train_pre,roll_sample[1],axis=0),
                                        # np.roll(train_caption_pad,roll_sample[2],axis=0),
                                        # np.roll(train_caption_pad,roll_sample[3],axis=0),
                                        # np.roll(train_caption_pad,roll_sample[4],axis=0),))
                                        ))
        # false_mfcc = np.concatenate((train_data_pad,
        #                              train_data_pad,
        #                             #  train_data_pad,
        #                             #  train_data_pad,
        #                             #  train_data_pad,))
        #                              ))

        false_pos = np.concatenate((train_pos,
                                    #  train_data_pad,
                                    #  train_data_pad,
                                    #  train_data_pad,
                                    #  train_data_pad,))
                                     ))
        # true_caption = train_caption_pad
        # true_mfcc = train_data_pad

        true_pre = train_pre
        true_pos = train_pos

        # ground_truth = [ 1 for _ in range(len(true_caption))] + [0 for _ in range(len(false_caption))]
        ground_truth = [ 1 for _ in range(len(true_pre))] + [0 for _ in range(len(false_pre))]
        # print(true_caption.shape)
        # print(false_caption.shape)
        # # print(ground_truth)
        # exit()
        # train_mfcc = np.concatenate((true_mfcc, np.array(false_mfcc)))
        # train_caption = np.concatenate((true_caption, np.array(false_caption)))

        train_pre = np.concatenate((true_pre, np.array(false_pre)))
        train_pos = np.concatenate((true_pos, np.array(false_pos)))

        total_sample_size = len(ground_truth)
        random_index = np.random.choice(total_sample_size,total_sample_size, replace=False)

        pre_input = train_pre[random_index]
        pos_input = train_pos[random_index]
        input_ground_truth = np.array(ground_truth)[random_index]

        hist = History()

        check_save  = ModelCheckpoint("model_other"+str(batch_size)+"_"+str(i)+str(epoch)+"{val_acc:.5f}.h5", period= 1)
        model.fit([pre_input, pos_input], input_ground_truth,
                  batch_size=batch_size,
                  validation_data = ([valid_pre, valid_pos],np.ones(len(valid_pre))),
                  epochs=1, callbacks=[check_save, hist])
