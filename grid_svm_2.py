import numpy as np
import scipy.io as io
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import time
#mat = io.loadmat("everything.mat")

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
    clist = np.logspace(0,1.7,3)
    gammas = np.logspace(-3.1,0,8)
    
    xmat = [np.zeros((130,650))] * 2447
    ymat = [np.zeros((130,))] * 2447
    
    arr = []
    #voxel_list = []
    seed = 104
    x_tr = X[0:104]
    y_tr = Y[0:104]
    x_te = X[104:130]
    y_te = Y[104:130]
    #(x_tr,x_te,y_tr,y_te) = train_test_split(X,Y,test_size=.2,random_state=104)
    for i in range(0,2447):
        xmat[i], ymat[i] = chooseV(i,x_tr,y_tr)

    best_acc_tot = []
    best_c_tot = []
    best_g_tot = []
    uniqs = []
    for vox in range(2447):
        print(vox)
        num_splits = 5
        kf = KFold(n_splits=num_splits,shuffle=True,random_state = seed)
        accs = np.zeros((3,8))
        my_list = []
        for train_idx,val_idx in kf.split(xmat[vox],ymat[vox]):
            x_t = xmat[vox][train_idx]
            y_t = ymat[vox][train_idx]
            x_v = xmat[vox][val_idx]
            y_v = ymat[vox][val_idx]

            jj = StandardScaler()
            x_t = jj.fit_transform(x_t)
            x_v = jj.transform(x_v)


            pi=0
            pj=0
            
            bsc=0
            bc=0
            bg=0
            for i in clist:
                pj=0
                for j in gammas:
                    clf = svm.SVC(C=i, gamma=j, kernel='rbf')
                    clf.fit(x_t,y_t)
                    score = clf.score(x_v,y_v)
                    #print(str(pi) + " " + str(pj) + " " + str(accs[pi][pj]))
                    accs[pi][pj]+=score
                    if(score>bsc):
                        bsc=score
                        bc=i
                        bg=j
                    pj+=1
                    
                pi+=1
            my_list.append((bc,bg))
        best_acc=0
        best_c=0
        best_g=0
        for i in range(3):
            for j in range(8):
                accs[i][j]/=num_splits
                if(accs[i][j]>best_acc):
                    best_acc=accs[i][j]
                    best_c=clist[i]
                    best_g=gammas[j]

        best_acc_tot.append(best_acc)
        best_c_tot.append(best_c)
        best_g_tot.append(best_g)

        uniqs.append(len(set(my_list)))
        
    
    junk = []
    junk.append(best_acc_tot)
    junk.append(best_c_tot)
    junk.append(best_g_tot)
    junk.append(uniqs)
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
        clf = svm.SVC(C=best_c_tot[i],gamma=best_g_tot[i],kernel='rbf')
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


