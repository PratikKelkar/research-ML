import numpy as np
from scipy.io import loadmat

def get_y():
    
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
    return returnfiley
