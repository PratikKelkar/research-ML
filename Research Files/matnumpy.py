import numpy as np
from scipy.io import loadmat

def mat_to_numpy():
    
    returnfilex = []
    returnfiley = np.zeros((110,1))

    words_in_cat = {"Tools": [11, 23, 35, 47, 59, 68], "Animals": [9, 21, 33, 45, 57]}
    allwords = [9, 11, 21, 23, 33, 35, 45, 47, 57, 59, 68]

    #create Y file - same for all bands:

    c1=0
    for segnum in range(1,11,1):
        for word in allwords:
            if word in words_in_cat["Tools"]:
                returnfiley[c1] = 0
            else:
                returnfiley[c1] = 1
            c1+=1

    #create list of X files -- len(5):

    mat_bands = []
    
    mat_bands.append(loadmat("delta.mat"))
    mat_bands.append(loadmat("theta.mat"))
    mat_bands.append(loadmat("alpha.mat"))
    mat_bands.append(loadmat("beta.mat"))
    mat_bands.append(loadmat("gamma.mat"))
    

    for i in mat_bands:
        numpy_x = np.zeros((110, 256, 650))      
        c2=0
        for segnum in range(1,11,1):
            for word in allwords:
                matkey = "W0" + str(word).zfill(2) + "_Segment_0" + str(segnum).zfill(2)
                numpy_x[c2] = i[matkey][:256, 200:850]
                c2+=1

        returnfilex.append(numpy_x)


    return (returnfilex, returnfiley)
