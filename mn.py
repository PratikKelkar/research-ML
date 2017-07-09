import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

xtrain = xtrain.reshape(xtrain.shape[0],28,28,1)
xtest = xtest.reshape(xtest.shape[0],28,28,1)
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain/=255
xtest/=255

ytrain = np_utils.to_categorical(ytrain,10)
ytest = np_utils.to_categorical(ytest,10)

model = Sequential() #start of model

model.add(Convolution2D(32,(3,3),input_shape=(28,28,1),activation="relu"))


model.add(Convolution2D(32,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(xtrain,ytrain,batch_size=32,nb_epoch=10,verbose=1)

score = model.evaluate(xtest,ytest,verbose=0)
print(score)
