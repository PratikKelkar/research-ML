import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.io as io


'''
x = np.array([[-1,-1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1,1,1,2,2,2])
'''

n = io.loadmat("juicy.mat")
arr = n["finwords"]
x = np.zeros((130,32500))
for i in range(130):
    a = arr[i][:,201:851]
    x[i] = a.flatten()

y = np.load("lessv_y.npy")


np.save("tokarthik_x.npy", x)
np.save("tokarthik_y.npy", y)
     
