import numpy as np


xt = np.zeros((1000,2))
yt = np.zeros((1000,1))
xtest = np.zeros((100,2))
ytest = np.zeros((100,1))
for i in range(0,1000):
    xt[i,0] = np.random.randint(0,64)
    xt[i,1] = np.random.randint(0,64)
    yt[i,0] = xt[i,0]+xt[i,1]

for i in range(0,100):
    xtest[i,0] = np.random.randint(0,1000)
    xtest[i,1] = np.random.randint(0,1000)
    ytest[i,0] = xtest[i,0]+xtest[i,1]

np.save("xt.npy",xt)
np.save("yt.npy",yt)
np.save("xtest.npy",xtest)
np.save("ytest.npy",ytest)
