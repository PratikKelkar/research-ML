import numpy as np
import scipy.io as io
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import time



#mat = io.loadmat("everything.mat")


X,Y = np.load("july6x.npy"),np.load("july6y.npy")


def chooseV(v,matx,maty):
    choicex = np.zeros((130, 650))

    c=0
    for i in matx:
        choicex[c] = i[v]
        c+=1

    return (choicex, maty)    

xmat = [np.zeros((130,650))]*2447
ymat = [np.zeros((130,))]*2447

for i in range(2447):
    xmat[i],ymat[i] = chooseV(i,X,Y)



karthu = np.load("chooseVoxels.npy")
set1 = karthu[0]


seed = 104


#print(set1.shape)

sortedset1 = sorted(set1, key=lambda x:x[0], reverse=True)


above60 = []



for i,j in sortedset1:
    #print(str(i) + " " + str(j))
    if(i>.6):
        above60.append(j)

to_sel=20
feats = np.zeros((130,0))

for i in range(to_sel):
    feats = np.concatenate((feats,xmat[int(above60[i])]),axis=1)

clist = np.logspace(-1,1.3,5)
#gammas = np.logspace(-6,2,12)
gammas = [1]
best_acc = 0
bestc=0
bestg=0
for i in clist:
    for j in gammas:
        num_splits=5
        kf = KFold(n_splits=num_splits,shuffle=True,random_state=seed)
        avg = 0
        for train_idx,test_idx in kf.split(feats,Y):
            clf = svm.SVC(C=i,gamma=j,kernel='linear')
            jj = StandardScaler()
            x_train = feats[train_idx]
            x_train = jj.fit_transform(x_train)
            x_test = jj.transform(feats[test_idx])
            y_train,y_test = Y[train_idx],Y[test_idx]
            clf.fit(x_train,y_train)
            avg+=clf.score(x_test,y_test)
        avg/=num_splits
        if(avg>best_acc):
            best_acc=avg
            bestc=i
            bestg=j
print(best_acc)
print(bestc)
print(bestg)
    
