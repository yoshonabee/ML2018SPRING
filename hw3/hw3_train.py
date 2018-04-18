import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Model, Input
from hw3_fc import *

print("\n======================================================Loading Training Data======================================================\n")
x_train, y_train = loadData(sys.argv[1], 'train')


train_generator = genImage(x_train, y_train)

print("\n==========================================================Building Models========================================================\n")
inputs = Input(shape=(48, 48, 1))
model1 = gen08(inputs)
model2 = gen09(inputs)
model3 = gen12(inputs)
model4 = func01(inputs)
model5 = func02(inputs)
model6 = Res02(inputs)
model7 = Res03(inputs)

print("\n==========================================================Compiling Models========================================================\n")
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model6.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model7.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epoch = 45
print("\n====================================================Start Training, Total 8 Models================================================\n")
print("\n===========================================================Training Model1========================================================\n")
model1.fit_generator(train_generator, steps_per_epoch=500, epochs=epoch)
print("\n===========================================================Training Model2========================================================\n")
model2.fit_generator(train_generator, steps_per_epoch=500, epochs=epoch)
print("\n===========================================================Training Model3========================================================\n")
model3.fit_generator(train_generator, steps_per_epoch=500, epochs=epoch)
print("\n===========================================================Training Model4========================================================\n")
model4.fit_generator(train_generator, steps_per_epoch=500, epochs=epoch)
print("\n===========================================================Training Model5========================================================\n")
model5.fit_generator(train_generator, steps_per_epoch=500, epochs=epoch)
print("\n===========================================================Training Model6========================================================\n")
model6.fit_generator(train_generator, steps_per_epoch=500, epochs=epoch)
print("\n===========================================================Training Model8========================================================\n")
model7.fit_generator(train_generator, steps_per_epoch=500, epochs=epoch)

print("\n==========================================================Ensembling Models=======================================================\n")
models = [model1, model2, model3, model4, model5, model6, model7]
model = ensemble(inputs, models)

print("\n============================================================Saving Model1=========================================================\n")
model.save('model.h5')
print("\n================================================================Done=========================================================\n")