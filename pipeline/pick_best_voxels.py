import numpy as np
import scipy.io as io
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

#import the other file that converts mat to numpy
from mat_to_numpy import mat_to_numpy

#get name of file as ParticipantName_Voxel/Channel
print("enter file name:")
matlab_matrix_name = str(input())
allX, allY = mat_to_numpy(matlab_matrix_name)

################## helphelphelphelphelphelphelphelphelp

#pick tools and animals with tools=0

X = np.zeros((130,2447,650))
Y = np.zeros((130,))


################## helphelphelphelphelphelphelphelphelp


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
# .25 <= c <= 256, num=11
# 1e-6 <= c <= 1, num=7
clist = np.logspace(-3, 7, base=2, num=11)
glist = np.logspace(-6, 0 , base=10, num=7)

#final lists to store all data
ALL_DATA = []
VOX_AVGS = []

#create names to save the final arrays as
savename = matlab_matrix_name[:-4]

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


        bestscore = 0
        bestc = 0
        bestg = 0

        #run svm
        for c in clist:
            for g in glist:
                clf = svm.SVC(C=c, gamma=g, kernel='linear',random_state=seed)
                clf.fit(x_t,y_t)
                score = clf.score(x_v,y_v)

                if score>bestscore:
                    bestscore = score
                    bestc = c
                    bestg = g

                ALL_DATA.append((vox, splitnum, c, g, score))

        voxel_avg += bestscore

 
    voxel_avg/=4
    VOX_AVG.append((vox, voxel_avg))

    #intermittently save the arrays in case program crashes
    if(vox!=0 and vox%300==0):
        np.save(savename+"_alldata.npy", ALL_DATA)
        np.save(savename+"_voxelavgs.npy", VOX_AVGS)
        
    print(str(vox) + " finished")

    
#Save final data
np.save(savename+"_alldata.npy", ALL_DATA)
np.save(savename+"_voxelavgs.npy", VOX_AVGS)

print("YAY IT WORKED")

