import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
clist = np.logspace(-10, 7, base=2)
#gammas = np.asarray([.0001,.001,.01,.1,1])
gammas = np.logspace(-17,0,base=2)
#store all data: vox#, split#, bestc, bestg, score
alldata = []


def run():

    TESTING = True
    
    
    for vox in range(2447):
        if(TESTING and (vox)%10==0 and vox!=0): vox+=40
        #split remaining data into training and validation using 5 fold kcross
        seed = 32
        num_splits = 5
                                                            #kf = KFold(n_splits=num_splits, shuffle=True, random_state=seed)
        vox_x, vox_y = chooseV(vox, x_train, y_train)

        #create the 5 cases to iterate over

        train_iters = [(vox_x[0:78], vox_y[0:78]),
                       (np.concatenate((vox_x[0:52],vox_x[78:104]), axis=0),np.concatenate((vox_y[0:52],vox_y[78:104]), axis=0)),
                        (np.concatenate((vox_x[0:26],vox_x[52:104]), axis=0),np.concatenate((vox_y[0:26],vox_y[52:104]), axis=0)),
                         (vox_x[26:104], vox_y[26:104])]

        valid_iters = [(vox_x[78:104], vox_y[78:104]),
                       (vox_x[52:78], vox_y[52:78]),
                       (vox_x[26:52], vox_y[26:52]),
                       (vox_x[0:26], vox_y[0:26])]
                        
        
        #run the cross-validation
        cplotlist = []
        gplotlist = []
        zlist = []
        for splitnum in range(4):

            x_t, y_t = train_iters[splitnum]
            x_v, y_v = valid_iters[splitnum]

            '''         
            #create training and validation sets for this split
            x_t = vox_x[train_idx]
            y_t = vox_y[train_idx]
            x_v = vox_x[vali_idx]
            y_v = vox_y[vali_idx]
            '''
                        
            #normalize
            jj = StandardScaler()
            x_t = jj.fit_transform(x_t)
            x_v = jj.transform(x_v)

            bestscore = 0
            bestc = 0
            bestg = 0
            
            #run gridsearch to find best c and gamma for this split
            for i in clist:
                for j in gammas:
                    clf = svm.SVC(C=i, gamma=j, kernel='rbf',random_state=seed)
                    clf.fit(x_t,y_t)
                    score = clf.score(x_v,y_v)
                    #print(clf.predict(x_v))
                    #if(i==1.0): print("LOL " + str(score))
                    cplotlist.append(i)
                    gplotlist.append(j)
                    zlist.append(score)
                    if score>bestscore:
                        bestscore = score
                        bestc = i
                        bestg = j

            #store all data
            alldata.append((vox, splitnum, bestc, bestg, bestscore))
            #if score>.6:
            print(str(vox) + ", split=" + str(splitnum+1) + ": bestc=" + str(bestc) + " , bestg=" + str(bestg) + " , score=" + str(bestscore))

            
            
            

run()



        
