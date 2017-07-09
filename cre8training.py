import numpy as np
import scipy.io as io

a = io.loadmat("C:/Users/pratik/Desktop/research/pratik_mat_smaller.mat")
#print(a["W067_Segment_001"].shape)


pcastle = np.zeros((10, 1590550))
pc = 0

pdrill = np.zeros((10, 1590550))
pd = 0

for key, val in a.items():
    if "W067_Segment" in key:
        pcastle[pc] = a[key][:,201:851].flatten()
        pc+=1
    elif "W068_Segment" in key:
        pdrill[pd] = a[key][:,201:851].flatten()
        pd+=1


castle = np.zeros((10, 397638))
drill = np.zeros((10, 397638))


#1590550
for x in range(397638):
    avg = (pcastle[:, 4*x] + pcastle[:, 4*x+1])
    if(x!=397637):
        avg += pcastle[:,4*x+2]+pcastle[:,4*x+3]
        avg /=4
    else:
        avg /=2
    #print(avg)
    castle[:,x] = avg
    
for x in range(397638):
    avg = (pdrill[:, 4*x] + pdrill[:, 4*x+1])
    if(x!=397637):
        avg += pdrill[:,4*x+2]+pdrill[:,4*x+3]
        avg /=4
    else:
        avg /=2
    #print(avg)
    drill[:,x] = avg
        

castledrill_x = np.zeros((20, 397638))
castledrill_y = np.zeros((20, 1))

for i in range(10):
    castledrill_x[i] = castle[i]
    castledrill_y[i] = 1

for i in range(10,20):
    castledrill_x[i] = drill[i-10]
    castledrill_y[i] = 2

np.save("castledrill_x.npy", castledrill_x)
np.save("castledrill_y.npy", castledrill_y)
