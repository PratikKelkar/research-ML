import numpy as np
#from keras.models import Sequential
#from keras.layers import Dropout, Activation, Dense
#from sklearn.model_selection import train_test_split
#from keras import metrics
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
from sklearn import svm
#from sklearn.externals import joblib

Xi = np.load('icax.npy')
Yi = np.load('icay.npy')
Yi.resize((130,))

seed = 10
#x_train,x_test,y_train,y_test = train_test_split(Xi,Yi,test_size=.2,random_state=seed)


jj = StandardScaler()
Xi = jj.fit_transform(Xi)


#clist = [.001,.01]
#gammas = [1e-5,1e-4]
clist = np.logspace(-3,2,6)
gammas = np.logspace(-8,2,11)



num_splits=5
best_acc = 0
bestc = 0
bestg = 0
for i in clist:
    for j in gammas:
        skf = StratifiedKFold(n_splits=num_splits,shuffle=True,random_state=425)
        avg = 0
        for train_idx,test_idx in skf.split(Xi,Yi):
             clf = svm.SVC(C=i,gamma=j,kernel='rbf')
             x_train,x_test = Xi[train_idx],Xi[test_idx]
             y_train,y_test = Yi[train_idx],Yi[test_idx]
             clf.fit(x_train,y_train)
             avg+=clf.score(x_test,y_test)

        avg/=num_splits
        if(avg>best_acc):
            best_acc=avg
            bestc = i
            bestg = j
        print(str(i)+" " + str(j) + " " + str(best_acc))

print(str(bestc) + " " + str(bestg))
print(best_acc)

