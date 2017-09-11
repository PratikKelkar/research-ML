import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from heapq import heappush, heappop, heappushpop
import itertools
from scipy.stats import pearsonr
from get_y import get_y
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
def finalsvm(totalwords, cat1, cat2):
    np.random.seed(49)
    x = np.load('svm_final_train.npy')
    y = get_y(cat1,cat2)
    jj = 10*totalwords #amount of total data
    tro = int(.8*jj) #amount of training data
    print(x.shape)
    print(y.shape)
    train_x = x[:tro]
    train_y = y[:tro]
    test_x = x[tro:jj]
    test_y = y[tro:jj]

    #clist = np.logspace(-3,8,1000)
    #clist = [.1,1]
    #print(clist)
    bst_acc = 0
    bst_c = 0
    '''
    for c in clist:
        avg_acc = 0
        for fold in range(4):
            kk = int(tro/4) #size of each fold
            valid_x = x[(fold*kk):((fold+1)*kk)]
            valid_y = y[(fold*kk):((fold+1)*kk)]

            tr2_x = np.concatenate((x[0:(fold*kk)],x[(fold+1)*kk:tro]),axis=0)
            tr2_y = np.concatenate((y[0:(fold*kk)],y[(fold+1)*kk:tro]),axis=0)

            scaler = StandardScaler()
            tr2_x = scaler.fit_transform(tr2_x)
            valid_x = scaler.transform(valid_x)
            tr2_y = np.ravel(tr2_y)
            valid_y = np.ravel(valid_y)

            classifier = LinearSVC(C=c)
            classifier.fit(tr2_x,tr2_y)
            avg_acc += (classifier.score(valid_x,valid_y))/4.0

        if(avg_acc>bst_acc):
            bst_acc = avg_acc
            bst_c = c
    '''
    
    
    
    bst_c = .01
    #print(str(bst_acc) + " " + str(bst_c))

    train_y = np.ravel(train_y)
    test_y = np.ravel(test_y)
    #now train and test on entirety

    model = Sequential()
    model.add(Dense(256,input_dim=300, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    model.fit(train_x,train_y,epochs=150,batch_size=30)
    print(model.evaluate(test_x,test_y))
    '''
    classifier = LinearSVC(C=bst_c)
    classifier.fit(train_x,train_y)
    
    print(classifier.score(test_x,test_y))
    print("Predicted: " + str(classifier.predict(test_x)))
    print("Actual   : " + str(test_y))
    '''
