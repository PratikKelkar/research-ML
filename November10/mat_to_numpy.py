import numpy as np
from scipy.io import loadmat

mat_bands = [loadmat("delta.mat"),
             loadmat("theta.mat"),
             loadmat("alpha.mat"),
             loadmat("beta.mat"),
             loadmat("gamma.mat")]
 
all_data = np.zeros( (5,63,10,256,650)) #remove beginning 200 ms
#all data is 5 bands x 68 words x 10 pres. x 256 channel x 650 ms
for band_num in range(len(mat_bands)):
    for word_num in range(6,69):
        for pres_num in range(10):
            entrykey = "W0" + str(word_num).zfill(2) + "_Segment_0" + str(pres_num+1).zfill(2) #add 1 for 1-indexing
            all_data[band_num][word_num-6][pres_num] = mat_bands[band_num][entrykey][:256,200:850]
 
           
np.save("all_data.npy",all_data)
 
words_in_category = np.full( (12,7), -1) #for every category, list of word indices
 
word_indexes = [ [11, 23, 35, 47, 59, 68], [9, 21, 33, 45, 57]
                    ,[12,24,36,48,60,66,67]
                    ,[6,18,30,42,54]
                    ,[7,19,31,43,55]
                    ,[8,20,32,44,56]
                    ,[10,22,34,46,58]
                    ,[13,25,37,49,61]
                    ,[14,26,38,50,62]
                    ,[15,27,39,51,63]
                    ,[16,28,40,52,64]
                    ,[17,29,41,53,65] ] #just a list version of words_in_category
#subtract 6 from everything
for i in range(len(word_indexes)):
  for j in range(len(word_indexes[i])):
    word_indexes[i][j] -= 6 #shifts
 
for i in range(len(word_indexes)):
    for j in range(len(word_indexes[i])):
        words_in_category[i][j] = word_indexes[i][j]
 
np.save("words_in_categories.npy",words_in_category)
 
lengths = np.zeros( (12,1))
for i in range(len(word_indexes)):
    lengths[i] = len(word_indexes[i])
 
np.save("category_lengths.npy",lengths)


