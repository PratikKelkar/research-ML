import numpy as np
from scipy.stats import pearsonr
from heapq import heappush, heappop, heappushpop
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools


f.write("-----")
g.write("-----")
np.random.seed(63)
all_data = np.load("all_data.npy") #holds all the data from channels
category_info = np.load("words_in_categories.npy") #category_info[cat][ptr] returns the number of the word(0...62) of the ptr'th word in the category cat
lengths = np.load("category_lengths.npy") #lengths[cat] is the number of words in category cat
tStart = 0 #start time
tEnd = 650 #end time
tWidth = 100 #width of time slice
tIncr = 50 #increment in start time
tEx = 10 #number of examples to downsample to

total_words = 63 

word_list = [] 

for i in range(63):
    word_list.append(i)

training_amt = 8 #8 examples for training, 2 for testing
testing_amt = 10 - training_amt
#build training table
TrainingData = np.zeros((5,total_words,training_amt,256,650))#gives the pertinent data from all_data for the two categories
TestingData = np.zeros( (5,total_words,testing_amt,256,650)) #^
wordptr = -1 #the index of the current word, iterates from 0...total_words


trainlabels= np.zeros((training_amt*total_words)) #0 or 1 for every training presentation
testlabels= np.zeros((testing_amt*total_words))#^ but for testing

testingWordNumLabels = np.zeros( (testing_amt*total_words)) #testingWordNumLabels[i] = the number of the word at the i'th presentation in testlabels
for i in word_list:
    wordptr+=1

    excl = [-1]*10 #excl[j] = the j'th presentation number which should be saved for testing (e.g. excl[0] = 0 means the first presentation of the wordptr'th word should be saved for testing). Ignore -1's.
    
    for pres in range(testing_amt):
        while(1): #this loop repeatedly generates a random presentation until one which hasn't been reserved for testing has been found, and then breaks it
            nxtrand = np.random.randint(0,10)
            if(excl[nxtrand]==-1):
                excl[nxtrand]=nxtrand
                break
    for bandnum in range(5):
        ptr2 = 0 #points to which presentation(0...9) of wordptr'th word we are currently copying to TrainingData
        for pres in range(10):
            if(excl[pres]!=-1): #if reserved for testing, don't include in training data
                continue
           
            TrainingData[bandnum][wordptr][ptr2]=all_data[bandnum][i][pres] #sets the channel x time matrix for TrainingData[bandnum][wordptr][ptr2]
            ptr2+=1 #move to next presentation

    for bandnum in range(5): #this loop is same as above, except now we only want the testing presentations
        ptr2=0
        for pres in range(10):
            if(excl[pres]==-1):
                continue
            TestingData[bandnum][wordptr][ptr2] = all_data[bandnum][i][excl[pres]]
            testingWordNumLabels[wordptr] = i #don't forget to save the word number of this presentation
            ptr2+=1
            
SHheap = [] #heap of BTC + featurevector information used to select top 400


f = open("word_results.txt","w")

for wordnum in range(total_words):
    for band_num in range(5): #frequency bands
        for t in range(tStart, tEnd-tWidth+1, tIncr): #various starts of time slice
            for channel in range(256): #eeg channels

                timeSequences = np.zeros( (training_amt,tEx) )
                
                for pres in range(training_amt):

                    trainlabels[pres]=categoryof[wordnum]
                    pos = 0
                    for tEStart in range(t,t+tWidth-tEx+1,tEx): #average pooling 10ms windows on the time segment
                        timeSequences[pres][pos] = np.average(TrainingData[band_num][wordnum][pres][channel][tEStart:tEStart+int(tWidth/tEx)])
                        pos+=1

                #pairwise correlations
                avg_p = 0
                for i in range(training_amt-1):
                    for j in range(i+1,training_amt):
                        avg_p += pearsonr(timeSequences[i],timeSequences[j])[0]
                avg_p /= training_amt*(training_amt-1)/2
                
                if(len(SHheap)<400):
                    heappush(SHheap, (avg_p,band_num,t,channel))
                else:
                    heappushpop(SHheap, (avg_p,band_num,t,channel))
    #pick top 5
    toSelect = 5
    f.write("Word " + str(wordnum) +"\n")
    print("Word " + str(wordnum))
    for i in range(400):
        (avg_p,band_num,t,channel) = heappop(SHheap)
        if(i>=400-toSelect):
            #this is da guy
            f.write(str(400-i+1) + ". " + str(band_num) + "   " + str(t) + "   " + str(channel) + "   " + str(avg_p) + "\n")
            print(str(400-i+1) + ". " + str(band_num) + "   " + str(t) + "   " + str(channel) + "   " + str(avg_p))
f.close()
print("finished!")
