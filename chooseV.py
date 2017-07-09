import numpy as np

karthu = np.load("chooseVoxels.npy")

set1 = karthu[0]
set2 = karthu[1]

#print(set1.shape)

sortedset1 = sorted(set1, key=lambda x:x[0], reverse=True)
sortedset2 = sorted(set2, key=lambda x:x[0], reverse=True)

above60one = []
above60two = []


for i,j in sortedset1:
    if(i>.6):
        above60one.append(j)

for i,j in sortedset2:
    if(i>.6):
        above60two.append(j)


