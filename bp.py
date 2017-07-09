import numpy as np
np.random.seed(7)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

allx = np.load("lessv_x.npy")
ally = np.load("lessv_y.npy")

xtrain = allx[0:120,:]
ytrain = ally[0:120,0]
xtest = allx[120:130,:]
ytest = ally[120:130,0]

print(xtrain.shape)

model = Sequential()
model.add(Dense(64,input_dim=10001,activation='relu'))
#model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer = 'rmsprop',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=10,batch_size=8)
score = model.evaluate(xtest,ytest,batch_size=8)
print(score)    
    

