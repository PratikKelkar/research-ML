import numpy as np

karthu = np.load("chooseVoxels.npy")

set1 = karthu[0]


#print(set1.shape)

sortedset1 = sorted(set1, key=lambda x:x[0], reverse=True)


above60 = []



for i,j in sortedset1:
    print(str(i) + " " + str(j))
    if(i>.6):
        above60.append(j)


