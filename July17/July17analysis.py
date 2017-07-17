import numpy as np

alldata = np.load("july17_alldata.npy")
voxavgs = np.load("july17_voxelavgs.npy")

avgslist = []
c=0
for i in range(2447):
    if(voxavgs[i,1]>.59):
        c+=1
    avgslist.append((voxavgs[i,0], voxavgs[i,1]))

print(c)
sortedavgs = sorted(avgslist, key=lambda tup: tup[1], reverse=True)


f = open("July17_accuracies.txt", "w")

f.write("###Voxels where average accuracy is 59% or greater \n")
f.write("###Total Voxels = 209 \n")
f.write("###Parameters used: c=3, gamma=0.005 \n")
f.write("\n")
f.write("Voxel#\t\tAccuracy \n\n")

for i in sortedavgs:
    vox,acc = i
    vox = int(vox)
    acc*=100
    if(acc>59):
        f.write(str(vox) + "\t\t" + str('{0:.2f}'.format(acc)))
        f.write("\n")

f.close()
