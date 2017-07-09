from sklearn.decomposition import FastICA as ica
import scipy.io as io
import numpy as np

mat = io.loadmat("beta.mat")
m = mat["words"]
n = np.zeros((130, 50, 950))

for i in range(130):
    yeet=ica(n_components=50, max_iter=1000,tol=0.0005).fit_transform(m[i].T)
    n[i] = yeet.T
    print(i)


fin = np.zeros((130, 47500))
for i in range(130):
    fin[i] = n[i].flatten()

print(fin.shape)

jj = np.load('tokarthik_y.npy')

np.save('icax.npy',fin)
np.save('icay.npy',jj)
