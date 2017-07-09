import numpy as np
import scipy.io as io
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import time
mat = io.loadmat("everything.mat")

#remove avg from all files

X,Y = np.load("july6x.npy"),np.load("july6y.npy")

#create matrix with the voxel
def chooseV(v, matx, maty): #v=voxelnumber
   
    choicex = np.zeros((matx.shape[0], 650))

    c=0
    for i in matx:
        choicex[c] = i[v]
        c+=1

    return (choicex, maty)    

def run():
    arr = []
    clist = np.logspace(0,1.7,2)
    gammas = np.logspace(-3.1,-1.8,4)
    
    xmat = [np.zeros((130,650))] * 2447
    ymat = [np.zeros((130,))] * 2447
    
    arr = []
    #voxel_list = []
    seed = 104
    (x_tr,x_te,y_tr,y_te) = train_test_split(X,Y,test_size=.2,random_state=104)
    for i in range(0,2447):
        xmat[i], ymat[i] = chooseV(i,x_tr,y_tr)
    

    best_c = [0]*2447
    best_g = [0]*2447
    best_acc = [0]*2447
    for i in clist:
        for j in gammas:

            marr = []
            print("YEET " + str(i) + " " + str(j))
            
            for k in range(0,2447):
                #ct = time.time()
                x,y = xmat[k],ymat[k]
                jj = StandardScaler()
                x = jj.fit_transform(x)
                num_splits = 5
                kf = KFold(n_splits=num_splits,shuffle=True,random_state=seed)
                avg = 0
                for train_idx,test_idx in kf.split(x,y):
                    clf = svm.SVC(C=i,gamma=j,kernel='rbf')
                    x_train,x_test = x[train_idx],x[test_idx]
                    y_train,y_test=y[train_idx],y[test_idx]
                    clf.fit(x_train,y_train)
                    avg+=clf.score(x_test,y_test)
                avg/=num_splits
                if(avg>best_acc[k]):
                    best_acc[k]=avg
                    best_c[k]=i
                    best_g[k]=j
                marr.append((avg,k))
                #print(nxct-ct)

            #arr.append(marr)
            #print(voxel_list)        

    
    junk = []
    junk.append(best_acc)
    junk.append(best_c)
    junk.append(best_g)
    asnp = np.asarray(junk)
    np.save("VoxelStuff.npy",junk)

    #run them all on testing data
    test_acc = [0]*2447


    
    for i in range(2447):
        jj = StandardScaler()
        mytrx = jj.fit_transform(xmat[i])
        mytry = ymat[i]
        tmp = chooseV(i,x_te,y_te)
        mytex = jj.transform(tmp[0])
        mytey =tmp[1]
        clf = svm.SVC(C=best_c[i],gamma=best_g[i],kernel='rbf')
        clf.fit(mytrx,mytry)
        scr = clf.score(mytex,mytey)
        test_acc[i]=(scr,i)
        if(scr>.6):
            print(str(i) + " " + str(scr))
    np.save("TestAccs.npy",np.asarray(test_acc))
    
    return np.asarray(arr)  
    #arr.append((i, acc))
    #print(acc)
    #return arr
      
