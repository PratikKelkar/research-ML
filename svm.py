import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn import metrics
from sklearn import grid_search
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
x = np.load('tokarthik_x.npy')
y = np.load('tokarthik_y.npy')
y.resize((130,))
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=.3,random_state=0)


'''
clist = [1e-5,1e-4,.001,.01,.1,1,10]
gammas = [1e-5,1e-4,.001,.01,.1,1,10]

#clist = [1e-9,1e-8,1e-7
#gammas = [.000001,.00001,.0001,.001]
param_grid = {'C':clist,'gamma':gammas}
clf = GridSearchCV(svm.SVC(kernel='rbf'),param_grid,cv=5)


print("1")
clf.fit(x_train,y_train)

joblib.dump(clf,'savemodel.pkl')

'''

clf = joblib.load('savemodel.pkl')


print(y_test)
print(clf.predict(x_test))
print(clf.score(x_test,y_test))
#predicted = cross_val_predict(clf,x_test,y_test,cv=3)
#print(metrics.accuracy_score(y_test,predicted))


