import numpy as np
import scipy.io as io
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import time
#mat = io.loadmat("everything.mat")


X,Y = np.load("july6x.npy"),np.load("july6y.npy")

#create matrix with the voxel
def chooseV(v, matx, maty): #v=voxelnumber
   
    choicex = np.zeros((130, 650))

    c=0
    for i in matx:
        choicex[c] = i[v]
        c+=1

    return (choicex, maty)    



def run():
    arr = []
    #clist = np.logspace(0,1.7,4)
    #gammas = np.logspace(-3.1,-1.8,4)

    pasdf = [(3.68694506452,0.0158489319246)]
    
    xmat = [np.zeros((130,650))] * 2447
    ymat = [np.zeros((130,))] * 2447

    for i in range(0,2447):
        #print(i)
        xmat[i], ymat[i] = chooseV(i,X,Y)

    
    #for i in clist:
     #   for j in gammas:


    voxels_to_pick = 10
    
    for ro in pasdf:
        i=ro[0]
        j=ro[1]

        print("YEET " + str(i) + " " + str(j))



        picked = [False]*2447
        current_features = np.zeros((130,0))
        voxel_list = []
        ct = time.time()

        last_acc=0
        for iteration in range(voxels_to_pick):
            
            best_acc_curr=0
            best_vox_curr=0
            curr_mine = []
            for k in range(0,2447):
                #ct = time.time()
                if(picked[k]):
                    continue
                
                seed = 104
                tmpt = xmat[k]
                #jj = StandardScaler()
                #tmpt = jj.fit_transform(tmpt)
                
                x,y = np.concatenate((current_features,tmpt),axis=1),ymat[k]
                num_splits =5
                kf = KFold(n_splits=num_splits,shuffle=True,random_state=seed)
                avg = 0
                for train_idx,test_idx in kf.split(x,y):
                    clf = svm.SVC(C=i,gamma=j,kernel='rbf')
                    jj = StandardScaler()
                    x_train = jj.fit_transform(x[train_idx])
                    x_test = jj.transform(x[test_idx])
                    y_train,y_test=y[train_idx],y[test_idx]
                    clf.fit(x_train,y_train)
                    avg+=clf.score(x_test,y_test)
                avg/=num_splits
                if(avg>best_acc_curr):
                    best_acc_curr=avg
                    best_vox_curr=k
                    curr_mine = x
                #print(nxct-ct)
            
            voxel_list.append(best_vox_curr)
            picked[best_vox_curr]=True
            last_acc=best_acc_curr
            current_features=np.concatenate((current_features,curr_mine),axis=1)
            print(best_acc_curr)
            print(voxel_list)
        nt = time.time()
        print("TIME TAKEN: " + str(nt-ct))
        print(voxel_list)
        print(last_acc)
    
    return arr  
    #arr.append((i, acc))
    #print(acc)
    #return arr
      
