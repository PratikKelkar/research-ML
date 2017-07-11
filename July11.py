import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

#load all data
X = np.load("july11x.npy")
Y = np.load("july11y.npy")

#split data into testing and training/validation
x_train = X[0:104]
y_train = Y[0:104]
x_test = X[104:130]
y_test = Y[104:130]

#creates a 130x650 matrix using only the specified voxel# v
def chooseV(v, matx, maty):
    choicex = np.zeros((matx.shape[0], 650))
    c=0
    for i in matx:
        choicex[c] = i[v]
        c+=1
    return (choicex, maty)

#create values of c and gamma for the gridsearch
clist = np.asarray([.25,1,4,16,64])
gammas = np.logspace(-6,0, base=10, num=4)

#store all data: vox#, split#, bestc, bestg, score
alldata = []

for vox in range(2447):
    #split remaining data into training and validation using 5 fold kcross
    seed = 64
    num_splits = 5
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=seed)

    vox_x, vox_y = chooseV(vox, x_train, y_train)

    splitnum=1
    for train_idx, vali_idx in kf.split(vox_x, vox_y):
        #create training and validation sets for this split
        x_t = vox_x[train_idx]
        y_t = vox_y[train_idx]
        x_v = vox_x[vali_idx]
        y_v = vox_y[vali_idx]
        
        #normalize
        jj = StandardScaler()
        x_t = jj.fit_transform(x_t)
        x_y = jj.transform(x_v)

        bestscore = 0
        bestc = 0
        bestg = 0

        #run gridsearch to find best c and gamma for this split
        for i in clist:
            for j in gammas:
                clf = svm.SVC(C=i, gamma=j, kernel='rbf')
                clf.fit(x_t,y_t)
                score = clf.score(x_y,y_v)
                if score>bestscore:
                    bestscore = score
                    bestc = i
                    bestg = j

        #store all data
        alldata.append((vox, splitnum, bestc, bestg, score))
        if score>.6:
            print(str(vox) + ", split=" + str(splitnum) + ": bestc=" + str(bestc)
                  + " , bestg=" + str(bestg) + " , score=" + str(score))
        else:
            print("YEET")
        
        splitnum+=1
                
        






        
