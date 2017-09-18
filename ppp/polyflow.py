from aug21svm import aug21svm
from aug21 import aug21
from aug21both import aug21both
from create_svm_data import create_svm_data
import numpy as np

from finalsvm import finalsvm
from finalsvmrbf import finalsvmrbf


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

#aug21(11,"Tools","Animals")
#create_svm_data(11)
#finalsvm(11,"Tools","Animals")
wordlist = ["Tools","Animals","Buildings","Body Parts","Furniture",
            "Vehicles","Kitchen Utensils",
            "Building Parts",
            "Clothing",
            "Insects",
            "Vegetables",
            "Man-made Objects"]
jjo = []
for i in range(100):
    jjo.append(0)

for c1idx in range(len(wordlist)):
    for c2idx in range(c1idx+1,len(wordlist)):
        
        cat1 = wordlist[c1idx]
        cat2 = wordlist[c2idx]
        

        lengtho = len(words_in_cat[cat1]) + len(words_in_cat[cat2])

        print(cat1 + " " + cat2)
        aug21(lengtho,cat1,cat2)
        create_svm_data(lengtho)
        print(cat1 + " " + cat2)
        fsd = finalsvm(lengtho,cat1,cat2)
        for gucci in range(100):
            jjo[gucci]+=fsd[gucci]

for i in range(len(jjo)):
    jjo[i]/=66
space = np.logspace(-3,2,100)
for i in range(100):
    print(str(space[i]) + ": " + str(jjo[i])) 

