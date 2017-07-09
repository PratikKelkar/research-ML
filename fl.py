#from keras.layers import Flatten
import scipy.io as io
import numpy as np

mat = io.loadmat('pratik_mat_smaller.mat')

print(mat['W011_Segment_002'][:,201:851].flatten().shape)

'''
counter = 0
for key, value in mat.items():
    if "W" in key:
        x = value[150,:]
        print(x.shape)
        mat[key] = x.flatten()
        print(mat[key].shape)
        counter += 1

print(mat['W011_Segment_002'].shape)
'''
