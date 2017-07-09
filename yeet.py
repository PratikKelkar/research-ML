import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.datasets import boston_housing

(xtrain,ytrain),(xtest,ytest) = boston_housing.load_data()
model = Sequential()
model.add(Dense(13,input_dim=13,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(13,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1,activation='linear'))

model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=10,batch_size=32)
score = model.evaluate(xtest,ytest,batch_size=16)
print(score)
