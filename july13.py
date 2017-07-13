import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate

data = np.load("june13alldata.npy")
above60 = []


for (a,b,c,d,e) in data:
    if e>.6:
        above60.append((a,b,c,d,e))

aa = {}
cs = []
gs = []
n60s = []
for i in above60:
    c = i[2]
    g = i[3]
    acc = i[4]
    if(acc>=.6):
        if((c,g) in aa):
           aa[(c,g)]+=1
        else:
            aa[(c,g)]=1

for key,val in aa.items():
    c,g=key
    cs.append(c)
    gs.append(g)
    n60s.append(val)

cnp = np.asarray(cs)
gnp = np.asarray(gs)
n60np = np.asarray(n60s)

xi,yi = np.linspace(cnp.min(),cnp.max(),100),np.linspace(gnp.min(),gnp.max(),100)
xi,yi = np.meshgrid(xi,yi)

rbf = scipy.interpolate.Rbf(cnp,gnp,n60np,function='linear')
zi = rbf(xi,yi)

plt.imshow(zi,vmin=n60np.min(),vmax=n60np.max(),origin='lower',
           extent=[cnp.min(),cnp.max(),gnp.min(),gnp.max()])
plt.scatter(cnp,gnp,c=n60np)
plt.colorbar()
plt.show()
