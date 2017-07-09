import scipy.io as io
import numpy as np

n = io.loadmat("juicy.mat")
kevin = n["finwords"]
y = np.reshape(np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), [130, 1])

myl = []
#print(y[10,0])
for i in range(130):
    a = kevin[i]
    b = a[:,400:600]
    #b.append(y[i,0])
    tlist = []
    for j in range(400,600):
        for vox in range(0,50):
            tlist.append(a[vox,j])
    
    tlist.append(y[i,0])
    myl.append(np.asarray(tlist))
    #myl.append(y[i,0])
   # if(i==0):
    #    print(len(myl))
    '''
    b = a[:, 400]
    myl.append(b.flatten())
    b = a[:,500]
    myl.append(b.flatten())
    b = a[:,600]
    myl.append(b.flatten())
    '''

karthu = np.asarray(myl)
karthu = np.reshape(karthu, [130, 10001])


#for i in range(130):
   # karthu[i,10000]=y[i]

newy = np.zeros((130, 1))
for i in range(130):
    newy[i,0] = y[i]

np.save("lessv_x.npy",karthu)
np.save("lessv_y.npy", newy)

