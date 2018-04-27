import os
import sys
import numpy as np
import tensorflow
from keras.models import Model, load_model
from keras.layers import *
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Conv2D
from hw3_fc import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def ensemble(inputs, models):
	outputs = [model.outputs[0] for model in models]
	average = Average()(outputs)
	model = Model(inputs=inputs, outputs=average)
	return model

#These functions of the models are defined in hw3_fc.py
#It is in the same folder

inputs = Input(shape=(48, 48, 1))
model1 = gen08(inputs)
model2 = gen09(inputs)
model3 = gen12(inputs)
model4 = func01(inputs)
model5 = func02(inputs)
model6 = Res02(inputs)
model7 = Res03(inputs)

print("=========================Loading weight========================")

model1.load_weights('weights/gen08.hd5f')
model2.load_weights('weights/gen09.hd5f')
model3.load_weights('weights/gen12.hd5f')
model4.load_weights('weights/func01.hd5f')
model5.load_weights('weights/func02.hd5f')
model6.load_weights('weights/ResNet02.hd5f')
model7.load_weights('weights/ResNet03.hd5f')


print("=============================Merging===========================")

models = [model1, model2, model3, model4, model5, model6, model7]
model = ensemble(inputs, models)
model.save('ensemble.h5')
print("============================Done================================")