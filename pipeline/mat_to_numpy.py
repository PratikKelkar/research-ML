import scipy.io as io
import numpy as np

#initial matlab matrix name should be NameofParticipant_Voxel/Channel.mat

matlab_matrix_name = ""#something
data = io.loadmat(matlab_matrix_name)

#row i is every single presentation
#column j is every category
# (i,j) = 1 if that word i is in category j
# one category per row

entire_matrix_x = np.zeros((630,2447,650))
entire_matrix_y = np.zeros((630,12))

cats = ['"Body Parts','Furniture','Vehicles','Animals',
        'Kitchen Utensils','Tools','Buildings','Building Parts',
        'Clothing','Insects','Vegetables','Man-made objects']

words_in_cat = [[6, 18, 30, 42, 54], [7, 19, 31, 43, 55], [8, 20, 32, 44, 56],
                [9, 21, 33, 45, 57], [10, 22, 34, 46, 58], [11, 23, 35, 47, 59, 68],
                [12, 24, 36, 48, 60, 66, 67], [13, 25, 37, 49, 61], [14, 26, 38, 50, 62],
                [15, 27, 39, 51, 63], [16, 28, 40, 52, 64], [17, 29, 41, 53, 65]]

i = 0
for key, val in data.items():
    if "W" in key:
        #add to entire_matrix_x
        entire_matrix_x[i] = val[:,201:851] 

        #generate y value and add to entire_matrix_y
        row_pres = np.zeros((12,))
        for cat in words_in_cat:
            for pres in cat:
                if str(key[2:4]) == str(pres).zfill(2):
                    row_pres[words_in_cat.index(cat)] = 1
        entire_matrix_y[i] = row_pres
        i+=1

savenamex = matlab_matrix_name[:-4] + "_x.npy"
savenamey = matlab_matrix_name[:-4] + "_y.npy"

np.save(savenamex, entire_matrix_x)
np.save(savenamey, entire_matrix_y)
            

