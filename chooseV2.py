import numpy as np
import scipy.io as io
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm

karthu = np.load("chooseVoxels.npy")
mat = io.loadmat("everything.mat")

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


X,Y = init()


def chooseV(v, matx, maty): #v=voxelnumber
   
    choicex = np.zeros((130, 650))

    c=0
    for i in matx:
        choicex[c] = i[v]
        c+=1

    return (choicex, maty)    




xmat = [np.zeros((130,650))]*2447
ymat = [np.zeros((130,))]*2447
for i in range(2447):
    xmat[i],ymat[i]=chooseV(i,X,Y)


seed = 104




set1 = karthu[0]
set2 = karthu[1]

#print(set1.shape)

sortedset1 = sorted(set1, key=lambda x:x[0], reverse=True)
sortedset2 = sorted(set2, key=lambda x:x[0], reverse=True)

above60one = []
above60two = []


for i,j in sortedset1:
    if(i>.6):
        above60one.append(j)

for i,j in sortedset2:
    if(i>.6):
        above60two.append(j)



pasdf = [(3.68694506452,0.0158489319246),(13.5935639088,0.00215443469003)]




to_sel = 1
curr_feat = np.zeros((130,0))
for i in range(to_sel):
    curr_feat = np.concatenate((curr_feat,xmat[int(above60one[i])]),axis=1)
    

#model 1
jj = StandardScaler()
curr_feat = jj.fit_transform(curr_feat)
num_splits = 5
seed = 104
kf = KFold(n_splits=num_splits,shuffle=True,random_state=seed)
avg = 0

for train_idx,test_idx in kf.split(curr_feat,Y):
    clf = svm.SVC(C=pasdf[0][0],gamma=pasdf[0][1],kernel='rbf')
    x_train,x_test=curr_feat[train_idx],curr_feat[test_idx]
    y_train,y_test=Y[train_idx],Y[test_idx]
    clf.fit(x_train,y_train)
    avg+=clf.score(x_test,y_test)
avg/=num_splits
print(avg)
    
