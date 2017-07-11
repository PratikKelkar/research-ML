import scipy.io
import numpy as np

mat = scipy.io.loadmat("everything.mat")

tbs = "11 12 23 24 35 36 47 48 59 60 66 67 68"
tbs = tbs.split(" ")

tools = "11 23 35 47 59 68".split(" ")

randomorder = {}
for i in tbs:
    temp = np.arange(1,11)
    np.random.shuffle(temp)
    randomorder[i] = temp

X = np.zeros((130,2447,650))
Y = np.zeros((130,))

xynum = 0
record=[]

for segn in range(10):
    for i in tbs:

        if i in tools:
            Y[xynum] = 1
            
        pres_name = "W0" + str(i) + "_Segment_0" + str(randomorder[i][segn]).zfill(2)
        pres_val = mat[pres_name]
        pres_val = pres_val[:,201:851]
        X[xynum]=pres_val
        xynum+=1

        record.append(pres_name)

        print(pres_name)
        print(pres_val.shape)
        print("====================")


np.save("July11x.npy", X)
np.save("July11y.npy", Y)


#all 13 words in a row, repeated ten times
#tools = 1, buildings = 0

