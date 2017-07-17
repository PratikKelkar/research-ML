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

#set values of c and gamma 
c = 3
g = 0.005

#final lists to store all data
ALL_DATA = []
TOTAL_AVGS = []

#loop through all 2447 voxels    
for vox in range(2447):
    seed = 32

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

    voxel_avg = 0

    for splitnum in range(4):

        x_t, y_t = train_iters[splitnum]
        x_v, y_v = valid_iters[splitnum]
                    
        #normalize
        jj = StandardScaler()
        x_t = jj.fit_transform(x_t)
        x_v = jj.transform(x_v)

        #run svm
        clf = svm.SVC(C=c, gamma=g, kernel='rbf',random_state=seed)
        clf.fit(x_t,y_t)
        score = clf.score(x_v,y_v)

        voxel_avg += score

        #store all data
        ALL_DATA.append((vox, splitnum, c, g, score))



    voxel_avg/=4
    TOTAL_AVG.append((vox, voxel_avg))

    #intermittently save the arrays in case program crashes
    if(vox!=0 and vox%300==0):
        np.save("july17_alldata.npy", ALL_DATA)
        np.save("july17_voxelavgs.npy", TOTAL_AVGS)
        
    print(str(vox) + " finished")

    
#Save final data
np.save("july17_alldata.npy", ALL_DATA)
np.save("july17_voxelavgs.npy", TOTAL_AVGS)

print("YAY IT WORKED")

