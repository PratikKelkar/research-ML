import scipy.io

n = scipy.io.loadmat("pratik_mat_smaller.mat")


for key, value in n.items():
    print(key)


