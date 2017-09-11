import numpy as np
from scipy.io import loadmat

def mat_to_numpy(category1, category2):
    
    returnfilex = []
    

    words_in_cat = {"Tools": [11, 23, 35, 47, 59, 68], "Animals": [9, 21, 33, 45, 57]
                    ,"Buildings":[12,24,36,48,60,66,67]
                    , "Body Parts":[6,18,30,42,54]
                    , "Furniture":[7,19,31,43,55]
                    , "Vehicles":[8,20,32,44,56]
                    , "Kitchen Utensils":[10,22,34,46,58]
                    , "Building Parts":[13,25,37,49,61]
                    , "Clothing":[14,26,38,50,62]
                    , "Insects":[15,27,39,51,63]
                    , "Vegetables":[16,28,40,52,64]
                    , "Man-made Objects":[17,29,41,53,65]}
    
    
    allwords = []
    for idx in words_in_cat[category1]:
        allwords.append(idx)
    for idx in words_in_cat[category2]:
        allwords.append(idx)
    returnfiley = np.zeros((10*len(allwords),1))
    #create Y file - same for all bands:

    c1=0
    for segnum in range(1,11,1):
        for word in allwords:
            if word in words_in_cat[category1]:
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
        numpy_x = np.zeros((10*len(allwords), 256, 650))      
        c2=0
        for segnum in range(1,11,1):
            for word in allwords:
                matkey = "W0" + str(word).zfill(2) + "_Segment_0" + str(segnum).zfill(2)
                numpy_x[c2] = i[matkey][:256, 200:850]
                c2+=1

        returnfilex.append(numpy_x)


    return (returnfilex, returnfiley)