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
print("mat_to_numpy done")

################## helphelphelphelphelphelphelphelphelp

cat_order = {"Body Parts":0,"Furniture":1,"Vehicles":2,"Animals":3,
             "Kitchen Utensils":4,"Tools":5,"Buildings":6,"Building Parts":7,
             "Clothing":8,"Insects":9,"Vegetables":10,"Man-made objects":11}


#pick tools and animals with tools=0

cat1X = []
cat2X = []
cat1Y = []
cat2Y = []

catTotX = []
catTotY = []
category_1_name = "Tools"
category_2_name = "Animals"

cat_1_idx = cat_order[category_1_name]
cat_2_idx = cat_order[category_2_name]
for i in range(630):
    if(allY[i,cat_1_idx]==1):
        catTotX.append(allX[i])
        catTotY.append(0) #cat1 represented as 0
    elif(allY[i,cat_2_idx]==1):
        catTotX.append(allX[i])
        catTotY.append(1) #cat2 represented as 1


sz = len(catTotX)

#X = np.zeros((sz,2447,650))
#Y = np.zeros((sz,))
X = np.asarray(catTotX)
Y = np.asarray(catTotY)

################## helphelphelphelphelphelphelphelphelp


#split data into testing and training/validation
test_amount = int(.2*sz)
train_amount = sz-test_amount
x_train = X[0:train_amount]
y_train = Y[0:train_amount]
x_test = X[train_amount:sz]
y_test = Y[train_amount:sz]

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
#clist = np.logspace(-3, 7, base=2, num=11)
#glist = np.logspace(-6, 0 , base=10, num=7)

clist = [x for x in np.logspace(-4, 10, base=2, num=30)]
#glist = [0.005]

#final lists to store all data
ALL_DATA = []
VOX_AVGS = []

#create names to save the final arrays as
savename = matlab_matrix_name[:-4]

#loop through all 2447 voxels

OVER60MATRIX = np.zeros((len(clist),))
CGAVGMATRIX = np.copy(OVER60MATRIX)


for c in clist:
    for vox in range(2447):

        print(str(vox) + " starting")


        AVGACC = 0
        
        seed = 32

        vox_x, vox_y = chooseV(vox, x_train, y_train)


        #create the 5 cases to iterate over
        train_iters=[]
        valid_iters=[]
        
        valid_amount = int(.2*train_amount)
        for i in range(5):
            val_st = i*valid_amount
            val_en = (i+1)*valid_amount
            valid_iters.append((vox_x[val_st:val_en],vox_y[val_st:val_en]))
            #now append train
            #0...val_st
            #val_en...train_amount
            train_iters.append((np.concatenate( (vox_x[0:val_st],vox_x[val_en:train_amount]),axis=0),
                                np.concatenate( (vox_y[0:val_st],vox_y[val_en:train_amount]),axis=0)))

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
            
            clf = svm.LinearSVC(C=c, random_state=seed)
            clf.fit(x_t,y_t)
            score = clf.score(x_v,y_v)

            AVGACC += score

            
           #ALL_DATA.append((vox, splitnum, c, g, score))

            #voxel_avg += bestscore

        AVGACC/=4
        if(AVGACC>0.6):
            OVER60MATRIX[clist.index(c)] +=1
            CGAVGMATRIX[clist.index(c)] += AVGACC

    print(str(clist.index(c))+"/"+str(len(clist)) + " done")
    print("===============================================")

CGAVGMATRIX_MYWAY = CGAVGMATRIX/2447
CGAVGMATRIX /= len(clist)

np.save("j28_meagan_over60.npy", OVER60MATRIX)
np.save("j28_meagan_cgavg_nielsen.npy", CGAVGMATRIX)
np.save("j28_meagan_cgavg_pratik.npy", CGAVGMATRIX_MYWAY)


print("YAY IT WORKED")
