import numpy as np
import scipy.io as io
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import time
mat = io.loadmat("everything.mat")

#remove avg from all files
def init():
    mat1 = {}
    for key,val in mat.items():
        if "W" in key:
            mat1[key] = val[:,201:851]
            #print(key)

    avg = np.zeros((2447, 650))
    for key,val in mat1.items():
        avg+=val
    avg/=130

    mat2 = {}
    for key,val in mat1.items():
        mat2[key] = val - avg

    rawvals = list(mat2.values())
    y = np.asarray([1,1,1,1,2,1,2,2,2,2,1,1,2,1,1,2,2,2,2,2,1,1,2,2,1,1,2,1,2,
         2,2,2,1,2,2,2,1,2,1,1,2,2,2,2,1,1,2,1,2,2,2,2,2,1,2,1,1,1,
         2,2,1,2,1,2,2,2,2,2,1,2,2,1,1,2,2,2,2,1,2,1,2,2,2,1,2,1,1,
         2,1,2,2,1,1,2,1,2,1,1,2,1,1,2,2,1,1,1,1,1,1,1,1,2,1,1,2,2,
         2,1,2,2,1,2,1,2,1,1,2,2,1,1])
    return (rawvals, y)

X = np.load("july6x.npy")
Y = np.load("july6y.npy")

#create matrix with the voxel
def chooseV(v, matx, maty): #v=voxelnumber
   
    choicex = np.zeros((130, 650))

    c=0
    for i in matx:
        choicex[c] = i[v]
        c+=1

    return (choicex, maty)    


def modelAcc(x,y):
    seed = 103
    jj = StandardScaler()
    x = jj.fit_transform(x)
 
    num_splits = 5
    best_acc = 0
    best_c = 0
    best_g = 0
    for i in clist:
        for j in gammas:
            kf = KFold(n_splits=num_splits,shuffle=True,random_state=seed)
            avg = 0
            for train_idx,test_idx in kf.split(x,y):
                clf = svm.SVC(C=i,gamma=j,kernel='rbf')
                x_train,x_test=x[train_idx],x[test_idx]
                y_train,y_test=y[train_idx],y[test_idx]
                clf.fit(x_train,y_train)
                avg+=clf.score(x_test,y_test)

            avg/=num_splits
            if(avg>best_acc):
                best_acc=avg
                best_c=i
                best_g=j

    print(str(best_c)+ " " + str(best_g))
    return best_acc

def run():
    arr = []
    clist = np.logspace(0,1.7,4)
    gammas = np.logspace(-3.1,-1.8,4)

    pasdf = [(3.68694506452,0.0158489319246)]
    
    xmat = [np.zeros((130,650))] * 2447
    ymat = [np.zeros((130,))] * 2447

    for i in range(0,2447):
        #print(i)
        xmat[i], ymat[i] = chooseV(i,X,Y)

    
    #for i in clist:
     #   for j in gammas:
    arr = []
    for ro in pasdf:
        i=ro[0]
        j=ro[1]
        marr = []
        print("YEET " + str(i) + " " + str(j))
        
        for k in range(0,2447):
            #ct = time.time()
            x,y = xmat[k],ymat[k]
            seed = 104
            #jj = StandardScaler()
            #x = jj.fit_transform(x)
            num_splits = 5
            kf = KFold(n_splits=num_splits,shuffle=True,random_state=seed)
            avg = 0
            for train_idx,test_idx in kf.split(x,y):
                clf = svm.SVC(C=i,gamma=j,kernel='rbf')
                jj = StandardScaler()
                x_train = x[train_idx]
                x_train=jj.fit_transform(x_train)
                x_test=jj.transform(x[test_idx])
                y_train,y_test=y[train_idx],y[test_idx]
                clf.fit(x_train,y_train)
                avg+=clf.score(x_test,y_test)
            avg/=num_splits
            marr.append((avg,k))
            #print(nxct-ct)

        arr.append(marr)
                
    return arr  
    #arr.append((i, acc))
    #print(acc)
    #return arr
      
tmp=run()
np.save('chooseVoxels.npy',np.asarray(tmp))
