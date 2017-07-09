import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from sklearn.model_selection import train_test_split
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
Xi = np.load('icax.npy')
Yi = np.load('icay.npy')
Yi.resize((130,))

x_train,x_test,y_train,y_test = train_test_split(Xi,Yi,test_size=.2,random_state=66)


jj = StandardScaler()
x_train = jj.fit_transform(x_train)
x_test = jj.transform(x_test)

model = Sequential()
model.add(Dense(units=64,input_dim=95000,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
model.fit(x_train,y_train,batch_size=8,epochs=10)
print(model.predict(x_test,batch_size=4))
print(y_test)
score = model.evaluate(x_test,y_test,batch_size=8,verbose=0)
print(score)

