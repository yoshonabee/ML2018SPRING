# -*- coding: UTF-8 -*-
import jieba
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys
import os
import pickle
from keras.preprocessing.text import Tokenizer

def splitChinese(inputStr, chWordSplit):
  import jieba
  if chWordSplit == True:
    import jieba
    seg_list = list(jieba.cut(inputStr, cut_all=False, HMM=True))
    splitString = " ".join(seg_list)
    return splitString
  else:
    result = ''
    for c in inputStr:
        result += c + ' '
    return result.strip()

def idx2weight(word_idx, vec_dim):
  unknow = 0
  weight_arr = np.zeros((len(word_idx) + 1, vec_dim)) # (all the word number)*(w2v dim)
  get = 0
  for word, i in word_idx.items(): # translate all the (word index) -> w2v
    if len(word_idx) + 1 == i:
      weight_arr[0] = w2v.wv[word]
      get += 1
    elif word_idx[word] is not None:
      weight_arr[i] = w2v.wv[word]
      get += 1
    else:
      unknow +=1
  print('unknow word:', unknow, ', get:', get)
  return weight_arr

def embedding(data, dim):
  emb_model = Word2Vec(data, size=dim, window=7, min_count=0, workers=12, sg=1, seed=0)
  print(emb_model.most_similar("沒"))
  print('embeding dim:', len(emb_model["沒"]))
  print('vocab size:', len(emb_model.wv.vocab))
  return emb_model

def readtest(testdir):
  import csv
  test_x = []
  with open(testdir, newline='') as csvfile:
    raw = csv.reader(csvfile, delimiter='\n', quotechar=',')
    for i, row in enumerate(raw):
      if i == 0:
        continue
      an5 = row[0].split('5:')[-1].replace('\t','')
      row[0] = row[0].split('5:')[0]
      an4 = row[0].split('4:')[-1].replace('\t','')
      row[0] = row[0].split('4:')[0]
      an3 = row[0].split('3:')[-1].replace('\t','')
      row[0] = row[0].split('3:')[0]
      an2 = row[0].split('2:')[-1].replace('\t','')
      row[0] = row[0].split('2:')[0]
      an1 = row[0].split('1:')[-1].replace('\t','')
      row[0] = row[0].split('1:')[0]
      an0 = row[0].split('0:')[-1].replace('\t','')
      row[0] = row[0].split('0:')[0]
      row[0] = row[0].split(',')[1].split('\t')
      # print(row[0],an0,an1,an2,an3,an4,an5)
      test_x.append([row[0],[an0,an1,an2,an3,an4,an5]])  

  return test_x

chsplit = False
# --------------train data load
rawX = []
y = []
from os import listdir
traindir = sys.argv[1]
if traindir[-1] != '/':
  traindir + '/'
ld = listdir(traindir)
for i, name in enumerate(ld): # ----from 1~5 file
  # if i != 3:
  with open(sys.argv[1] + name,'r') as trainfile:
    for row in trainfile:
      row = row.replace('\n','').lower()
      ch = splitChinese(row, chsplit)
      # print(splitChinese(row),len(ch))
      rawX.append(ch)
      y.append(i)
  
# --------------data preprocess --------------------------
vec_dim = 300
Xlist = [ word.split(" ") for word in rawX]
if os.path.exists('./w2vpca'):
  w2v = Word2Vec.load('./w2vpca')
else:
  w2v = embedding(Xlist, vec_dim)# FastText(Xlist, size=300, window=5, min_count=5, workers=12, sg=1, seed=0)
  w2v.save('./w2vpca')

print('got',len(Xlist),'data.')
#------------------token----------------
#tknzr = Tokenizer(filters='\t')
#tknzr.fit_on_texts(rawX)
if os.path.exists('./tkpca.pickle'):
  #w2v = Word2Vec.load('./w2vpca')
  with open('tkpca.pickle', 'rb') as tknfile:
    tknzr = pickle.load(tknfile)
else:
  tknzr = Tokenizer(filters='\t')
  tknzr.fit_on_texts(rawX)
  with open('tkpca.pickle', 'wb') as tknfile:
    pickle.dump(tknzr, tknfile)
word_seq = tknzr.texts_to_sequences(rawX)
word_idx = tknzr.word_index
#----------------------------------------------
print('rawX[0]:', rawX[0])
print('word_seq[0]:', word_seq[0])
print('word_idx["沒"]:', word_idx["沒"])
wordNum = len(word_idx)
print('got tokens:', wordNum)
embedding_weights = idx2weight(word_idx, vec_dim)
longestSent = 0
for i in word_seq:
  if len(i) > longestSent:
    longestSent = len(i)

print("longest sentence:", longestSent)

X = pad_sequences(word_seq, maxlen=longestSent)
'''
ex: 
longestSentence = 8
[0,0,0,0,10,218,100,12]
numbers n(int) in list is the sequence in embedding_weights[n(int)]
'''
# -----------data 2D to 1D long vector --------------------
'''
data 2D to 1D long vector：
shape(longestSentence*vectordim, )
'''
tensor = []
for row in X:
  # tmpten = embedding_weights[row[0]].copy()
  # print(len(row))
  tmp = embedding_weights[row[0]]*0
  for j in row:
    tmp += embedding_weights[j]
    # tmpten = np.concatenate((tmpten,embedding_weights[j]),axis=0)
  # tmpten = tmpten[vec_dim:]
  tensor.append(tmp)
  # print(np.amax(tmpten))
  # print(len(embedding_weights))

# X = np.array(tensor)
# print('X shape:', X.shape)
# -------------------------演算法----------------------
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
pca = PCA(n_components=250, whiten=True, svd_solver='arpack', random_state=0)
newData=pca.fit_transform(np.array(tensor))
print(newData)
kmean= KMeans(n_clusters=5, init='k-means++', max_iter=1000, verbose=1, random_state=33, n_jobs=4)
pt = kmean.fit(newData,y)


# -------------------------test data load---------------
test_x = readtest(sys.argv[2])
for i, row in enumerate(test_x):
  row[0] = [splitChinese(x.replace(' ',''), chsplit) for x in row[0]]
  row[0] = pad_sequences(tknzr.texts_to_sequences(row[0]), maxlen=longestSent) # to seq
  
  tensor = []
  # embed
  for r in row[0]:
    # tmp = embedding_weights[r[0]].copy()
    tmp = embedding_weights[r[0]]*0
    for ele in r:
      # tmp = np.concatenate((tmp,embedding_weights[ele]))
      tmp += embedding_weights[ele]
    # tmp = tmp[vec_dim:]
    tensor.append(tmp)

  row[0] = tensor 
  
  y_tensor = []
  for y in row[1]:
    y = splitChinese(y.replace(' ',''), chsplit)
    y_tensor.append(y)
  y_tensor = pad_sequences(tknzr.texts_to_sequences(y_tensor), maxlen=longestSent) # to seq
  row[1] = y_tensor

  y_tensor = []
  # embed
  for r in row[1]:
    # tmp = embedding_weights[r[0]].copy()
    tmp = embedding_weights[r[0]]*0
    # print(tmp)
    for ele in r:
      # tmp = np.concatenate((tmp,embedding_weights[ele]))
      tmp += embedding_weights[ele]
    # tmp = tmp[vec_dim:]
    y_tensor.append(tmp)
  
  row[1] = y_tensor
'''
結果test_x:
dtype: list
每ROW元素：[Qs, As] = [問題(list), 答案(list)]
Qs = [ndarray, ndarray, ...]
As = [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray] = [答案0,答案1,答案2,答案3,答案4,答案5]
'''

# print(test_x[0])
# print(test_x[1])
# exit()
  # print(y_tensor)
    # for ele_y in y:
      # print(ele_y)
# print(test_x[0])
  # exit()
# print(test_x[0])
# exit()
result = []
for row in test_x:
  subq = [0,0,0,0,0,0]
  for q in row[0]:
    # print('ans:',kmean.predict(pca.transform(q.reshape(1,-1))))
    for i in range(6):
      # print(kmean.transform(pca.transform(np.array(row[1][i]).reshape(1,-1))))
      # # print(kmean.fit_predict(pca.transform(np.array(row[1][i]).reshape(1,-1))))
      # # print(kmean.fit_transform(pca.transform(np.array(row[1][i]).reshape(1,-1))))
      # print(kmean.predict(pca.transform(np.array(row[1][i]).reshape(1,-1))))
      if kmean.predict(pca.transform(q.reshape(1,-1))) == kmean.predict(pca.transform(np.array(row[1][i]).reshape(1,-1))):
        # print(kmean.predict(pca.transform(np.array(row[1][i]).reshape(1,-1))))
        subq[i] += kmean.score(pca.transform(q.reshape(1,-1)))#1
  result.append(subq)
result = np.array(result)*(-1) + 0.5
print(len(result))

import csv
text = open('opca.csv', "w+")
s = csv.writer(text, delimiter=',',lineterminator='\n')
for i in result:
    s.writerow(i) 
text.close()

# print(np.amin result)

# seg_list = jieba.cut("我来到北京清华大学", cut_all=False, HMM=True)
# print("Default Mode: " + "/ ".join(seg_list))  # 默认模式