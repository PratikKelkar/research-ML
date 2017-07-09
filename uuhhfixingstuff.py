import numpy as np
import scipy.io as io


mat = io.loadmat("everything.mat")

def init():
    mat1 = {}
    for key,val in mat.items():
        if "W" in key:
            mat1[key] = val[:,201:851]

    avg = np.zeros((2447, 650))
    for key,val in mat1.items():
        avg+=val
    avg/=130

    mat2 = {}
    for key,val in mat1.items():
        mat2[key] = val - avg

    rawvals = list(mat2.values())
    rawkeys = list(mat2.keys())

    return (rawkeys, rawvals)

q,w = init()

w = np.asarray(w)
qy = []
for i in range(130):
    if "11" in q[i]:
        qy.append(1)
    elif "23" in q[i]:
        qy.append(1)
    elif "35" in q[i]:
        qy.append(1)
    elif "47" in q[i]:
        qy.append(1)
    elif "59" in q[i]:
        qy.append(1)
    elif "68" in q[i]:
        qy.append(1)
    else:
        qy.append(2)

qy = np.asarray(qy)

np.save("july6x.npy", w)
np.save("july6y", qy)
