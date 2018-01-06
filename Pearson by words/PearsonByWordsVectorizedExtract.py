import numpy as np
from scipy.stats import pearsonr
from heapq import heappush, heappop, heappushpop
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools
import pickle

run_num = 5
f = open("wordsresults_" + str(run_num)+".txt", "w")
#g = open("categ_results_"+str(run_num)+".txt","w")
f.write("-----")
#g.write("-----")
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
TrainingData = np.zeros((total_words,5,training_amt,256,650))#gives the pertinent data from all_data for the two categories
TestingData = np.zeros( (total_words,5,testing_amt,256,650)) #^
wordptr = -1 #the index of the current word, iterates from 0...total_words




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
           
            TrainingData[wordptr][bandnum][ptr2]=all_data[bandnum][i][pres] #sets the channel x time matrix for TrainingData[bandnum][wordptr][ptr2]
            ptr2+=1 #move to next presentation

    for bandnum in range(5): #this loop is same as above, except now we only want the testing presentations
        ptr2=0
        for pres in range(10):
            if(excl[pres]==-1):
                continue
            TestingData[wordptr][bandnum][ptr2] = all_data[bandnum][i][excl[pres]]
            ptr2+=1
            

print("yellow")

toSelect = 5

#todo: fix hstack stuff

best_feature_vectors = np.zeros( (total_words, training_amt,toSelect * tEx) )
test_feature_vectors = np.zeros( (total_words, testing_amt, toSelect * tEx) )

#5*12*256*8*10=

timeSequences = np.zeros( (total_words,5,12,training_amt,256,tEx) )
AverageWord = np.zeros( (total_words,5,12,256,tEx) )



fixedc = int(tWidth/tEx)
ptrr = 0
for t in range(tStart, tEnd-tWidth+1, tIncr):
    ptrppp = 0
    for tEStart in range(t,t+tWidth-tEx+1,tEx):
        timeSequences[:,:,ptrr,:,:,ptrppp] = np.average(TrainingData[:,:,:,:,tEStart:tEStart+fixedc], axis = 4)
        AverageWord[:,:,ptrr,:,ptrppp] = np.average(timeSequences[:,:,ptrr,:,:,ptrppp],axis=2)
        ptrppp+=1
    ptrr+=1
print(str(timeSequences.shape))

#interesting: see if each presentation of a word differs from the "Average word" significantly using pearson correlations
print("blue")
for wordnum in range(total_words):
    SHheap = [] #heap of BTC + featurevector information used to select top 400
    
    for band_num in range(5): #frequency bands
        ptrr=0
        for t in range(tStart, tEnd-tWidth+1, tIncr): #various starts of time slice
            for channel in range(256): #eeg channels

                #pairwise correlations
                avg_p = 0
                avg_p2 = 0
                #print(str(wordnum) + " " + str(band_num) + " " + str(ptrr) + " " + str(channel))
                for i in range(training_amt-1):
                    for j in range(i+1,training_amt):
                        #if(wordnum == 1):
                       #     print(str(pearsonr(timeSequences[wordnum][band_num][ptrr][channel][i],timeSequences[wordnum][band_num][ptrr][channel][j])))
                        avg_p += pearsonr(timeSequences[wordnum][band_num][ptrr][i][channel],timeSequences[wordnum][band_num][ptrr][j][channel])[0]

                '''
                for word2 in range(total_words):
                    if(wordnum==word2):
                        continue
                    avg_p2 += pearsonr(AverageWord[wordnum][band_num][ptrr][channel], AverageWord[word2][band_num][ptrr][channel])[0]
                '''
                avg_p /= training_amt*(training_amt-1)/2 #want to maximize
                #avg_p2 /= (total_words-1) #want to minimize
                #ranking_measure = (avg_p - avg_p2)/2 #want to maximize
                if(len(SHheap)<400):
                    heappush(SHheap, (avg_p,band_num,t,channel, timeSequences[wordnum,band_num,ptrr,:,channel]))
                else:
                    heappushpop(SHheap, (avg_p,band_num,t,channel, timeSequences[wordnum,band_num,ptrr,:,channel]))
            ptrr+=1
    #pick top 5
    
    f.write("Word " + str(wordnum) +"\n")
    print("Word " + str(wordnum))

    
    current_matrix = np.zeros( (training_amt,0))
    test_matrix = np.zeros( (testing_amt,0))
    
    for i in range(400):
        (avg_p,band_num,t,channel, timeSequenc) = heappop(SHheap)
        if(i>=400-toSelect):
            #this is da guy
            f.write(str(400-i) + ". " + str(band_num) + "   " + str(t) + "   " + str(channel) + "   " + str(avg_p) + "\n")
            print(str(400-i) + ". " + str(band_num) + "   " + str(t) + "   " + str(channel) + "   " + str(avg_p))
            current_matrix = np.hstack( (current_matrix,timeSequenc))

            #construct testing matrix
            tmpo = np.zeros( (testing_amt,tEx))
            for itero in range(testing_amt):
                pp = 0
                for tEStart in range(t,t+tWidth-tEx+1,tEx):
                    tmpo[itero][pp] = np.average(TestingData[wordnum,band_num,itero,channel,tEStart:tEStart+int(tWidth/tEx)])
                    pp+=1
            test_matrix = np.hstack( (test_matrix,tmpo) )
            
    best_feature_vectors[wordnum] = current_matrix
    test_feature_vectors[wordnum] = test_matrix

clist = np.logspace(-3,2,100)
avgacc=0
save_trainx = {}
save_trainy = {}
save_testx = {}
save_testy = {}
for cat1 in range(12):
    for cat2 in range(cat1+1,12):
        ptr = 0
        tot_words = int(lengths[cat1][0])+int(lengths[cat2][0])
        
        trainx = np.zeros ( ( training_amt * tot_words,toSelect * tEx))
        trainy = np.zeros( (training_amt * tot_words) )
        testx = np.zeros ( (testing_amt * tot_words, toSelect*tEx))
        testy = np.zeros( (testing_amt * tot_words))
        ptr2 = 0
        for itero in range(training_amt):
            while ptr<7 and category_info[cat1][ptr]!=-1:
                tword = category_info[cat1][ptr]
                
                trainx[ptr2] = best_feature_vectors[tword][itero]
                trainy[ptr2] = 0
                ptr2+=1
                
                ptr+=1
            ptr = 0
            while ptr < 7 and category_info[cat2][ptr]!=-1:
                tword = category_info[cat2][ptr]
                
                trainx[ptr2] = best_feature_vectors[tword][itero]
                trainy[ptr2] = 1
                ptr2+=1
                ptr+=1
        ptr = 0
        ptr2=0
        for itero in range(testing_amt):
            while ptr < 7 and category_info[cat1][ptr]!=-1:
                tword = category_info[cat1][ptr]
                testx[ptr2] = test_feature_vectors[tword][itero]
                testy[ptr2] = 0
                ptr2+=1
                ptr+=1
            ptr = 0
            while ptr < 7 and category_info[cat2][ptr]!=-1:
                tword = category_info[cat2][ptr]

                testx[ptr2] = test_feature_vectors[tword][itero]
                testy[ptr2] = 1
                ptr2+=1
                ptr+=1
            ptr = 0
        save_trainx[(cat1,cat2)] = trainx
        save_trainy[(cat1,cat2)] = trainy
        save_testx[(cat1,cat2)] = testx
        save_testy[(cat1,cat2)] = testy

        #run kfold cross validation for parameters
        bst_acc = 0
        bst_c = 0
        
        for c in clist:
            avg_acc = 0
            for fold in range(4):
                fold_sz = int(training_amt*tot_words/4)
                valid_x = trainx[(fold_sz*fold):((fold_sz)*(fold+1))]
                valid_y = trainy[(fold_sz*fold):((fold_sz)*(fold+1))]
                tr_x = np.concatenate( (trainx[0:(fold_sz*fold)],trainx[(fold+1)*fold_sz:(training_amt*tot_words)]), axis = 0)
                tr_y = np.concatenate( (trainy[0:(fold_sz*fold)],trainy[(fold+1)*fold_sz:(training_amt*tot_words)]), axis = 0)

                scaler = StandardScaler()
                tr_x = scaler.fit_transform(tr_x)
                valid_x = scaler.transform(valid_x)
                tr_y = np.ravel(tr_y)
                valid_y = np.ravel(valid_y)

                classifier = LinearSVC(C=c)
                classifier.fit(tr_x,tr_y)
                avg_acc += (classifier.score(valid_x,valid_y) )/4.0
            if(avg_acc > bst_acc):
                bst_acc = avg_acc
                bst_c = c
        print("For " + str(cat1) + " and " + str(cat2) + " we picked C = " + str(bst_c))
        classifier = LinearSVC(C=bst_c)
        scaler = StandardScaler()
        trainx = scaler.fit_transform(trainx)
        testx = scaler.transform(testx)
        trainy = np.ravel(trainy)
        testy = np.ravel(testy)
        classifier.fit(trainx, trainy)
        myscore = classifier.score(testx,testy)
        avgacc+=myscore
        print("Has accuracy " + str(myscore))
        print("======")
        
pickle.dump( (save_trainx,save_trainy,save_testx,save_testy), open("created_data.p","wb") )
avgacc/=(12*11/2)
print("Average score was " + str(avgacc))
f.close()


print("finished!")
