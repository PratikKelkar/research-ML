
#[(1125, 0.53846153846153844), (757, 0.53846153846153844), (2350, 0.53846153846153844), (1126, 0.53846153846153844), (1336, 0.53846153846153844), (758, 0.53846153846153844), (540, 0.53846153846153844), (783, 0.53846153846153844), (972, 0.53846153846153844), (1291, 0.53846153846153844), (894, 0.53846153846153844), (2301, 0.53846153846153844), (116, 0.53846153846153844), (506, 0.53846153846153844), (667, 0.53846153846153844), (782, 0.53846153846153844), (1508, 0.53846153846153844), (1708, 0.53846153846153844), (1845, 0.53846153846153844), (960, 0.53846153846153844), (1436, 0.53846153846153844)]



import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import random
#SEED


#load list of important voxels
f = open("July19_impvoxels.txt", "r")
important_vox = []
for i in f:
    if i!='2376': important_vox.append(int(i[:-1]))
important_vox.append(2376)
f.close()

prelimabove65 = [(1125, 0.72115384615384626), (757, 0.69230769230769229), (2350, 0.69230769230769229), (1126, 0.68269230769230771), (1336, 0.68269230769230771), (758, 0.6826923076923076), (540, 0.67307692307692313), (783, 0.67307692307692313), (972, 0.67307692307692313), (1291, 0.67307692307692313), (894, 0.66346153846153844), (2301, 0.66346153846153844), (116, 0.65384615384615385), (506, 0.65384615384615385), (667, 0.65384615384615385), (782, 0.65384615384615385), (1508, 0.65384615384615385), (1708, 0.65384615384615385), (1845, 0.65384615384615385), (960, 0.65384615384615374), (1436, 0.65384615384615374)]
above65 = [x for x,y in prelimabove65]

#load all data
X = np.load("july11x.npy")
Y = np.load("july11y.npy")

#split data into testing and training/validation
x_train = X[0:104]
y_train = Y[0:104]
x_test = X[104:130]
y_test = Y[104:130]

#creates a 130x650 matrix using only the specified voxel# v
def chooseV(v, matx):
    choicex = np.zeros((matx.shape[0], 650))
    c=0
    for i in matx:
        choicex[c] = i[v]
        c+=1
    return choicex

#set values of c and gamma 
c = 40
g = 0.00001


def feature_matrix(voxel_list):
  for i in voxel_list:
    current_vox = chooseV(i, x_train)
    #print(current_vox.shape)
    if voxel_list.index(i)==0: 
      feat_mat = np.copy(current_vox)
    else: 
      feat_mat = np.concatenate((feat_mat, current_vox), axis=1)
      
  return feat_mat    
    
      
      
    
def judge_voxels(voxel_list,c_param,g_param):
  
  seed = 32
  random.seed(seed)
	#generate feature matrix  
  vox_x = feature_matrix(voxel_list)
  vox_y = np.copy(y_train)
  
  #130 x (650*??)
  
  train_iters =  [(vox_x[0:78], vox_y[0:78]),
                     (np.concatenate((vox_x[0:52],vox_x[78:104]), axis=0),np.concatenate((vox_y[0:52],vox_y[78:104]), axis=0)),
                      (np.concatenate((vox_x[0:26],vox_x[52:104]), axis=0),np.concatenate((vox_y[0:26],vox_y[52:104]), axis=0)),
                       (vox_x[26:104], vox_y[26:104])]
  val_iters = [(vox_x[78:104], vox_y[78:104]),
                     (vox_x[52:78], vox_y[52:78]),
                     (vox_x[26:52], vox_y[26:52]),
                     (vox_x[0:26], vox_y[0:26])]

  my_avg = 0
  for splitnum in range(4):
    x_t,y_t = train_iters[splitnum]
    x_v,y_v = val_iters[splitnum]
    
    jj = StandardScaler()
    x_t = jj.fit_transform(x_t)
    x_v = jj.transform(x_v)
    
    clf = svm.SVC(C=c_param, gamma=g_param, kernel='rbf',random_state=seed)
    clf.fit(x_t,y_t)
    score = clf.score(x_v,y_v)
    my_avg += score
    #print(clf.predict(x_v))
  my_avg/=4
  return my_avg


mclist = [.01,.05,.1,.25,.5,1,3,8,12,16,24,32,50,64,86,100,128,150,200,256]
myglist = [1e-6,5e-6,1e-5,3e-5,5e-5,1e-4,3e-4,5e-4,7e-4,1e-3,3e-3,5e-3,8e-3,1e-2,3e-2,6e-2,.1,.5,1,2,5]


tuper = []
while(True):
  bstacc = 0
  bstc = 0
  bstg = 0
  iternum = 0
  for cr in mclist:
      for mg in myglist:
          iternum+=1
          current_base_acc = judge_voxels(above65,cr,mg)
          if(current_base_acc>bstacc):
              bstacc=current_base_acc
              bstc=cr
              bstg=mg
          print(str(iternum) + " done")
          print(current_base_acc)
          tuper.append((cr,mg,current_base_acc))
            
  print(str(bstacc) + " " + str(bstc) + " " + str(bstg))
          
  '''
  loopingaccs = []
  for vox in above65:
    tmp = [x for x in above65 if x!=vox]
    loopingaccs.append((vox, judge_voxels(tmp)))
  
  sortedloopingaccs = sorted(loopingaccs, key=lambda tu: tu[1], reverse=True)
  print(sortedloopingaccs)
  '''
  break
np.save("Param accuracies",np.asarray(tuper))

print("YAY IT WORKED")


