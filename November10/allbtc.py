import numpy as np
from scipy.stats import pearsonr
from heapq import heappush, heappop, heappushpop
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools

def get_feature_matrix(category1, category2):
    f.write("-----")
    
    np.random.seed(63)
    all_data = np.load("all_data.npy")
    category_info = np.load("words_in_categories.npy")
    lengths = np.load("category_lengths.npy")
    tStart = 0 #start time≈
    tEnd = 650 #end time
    tWidth = 100 #width of time slice
    tIncr = 50 #increment in start time
    tEx = 10 #number of examples to downsample to
 
    total_words = int(lengths[category1] + lengths[category2])
 
    #get the words in the categories
    word_list = []
 
    #get from category 1
    ptr = 0
    categoryof = [] #0 if category 1, 1 if category 2
    while ptr<7 and category_info[category1][ptr]!=-1:
        word_list.append(category_info[category1][ptr])
        categoryof.append(0)
        ptr+=1
    ptr = 0
    while ptr < 7 and category_info[category2][ptr]!=-1:
        categoryof.append(1)
        word_list.append(category_info[category2][ptr])
        ptr+=1

    training_amt = 8
    
    #build training table
    TrainingData = np.zeros((5,total_words,training_amt,256,650))
    TestingData = np.zeros( (5,total_words,10-training_amt,256,650))
    wordptr = -1 #points to 0...11 (total_words)


    trainlabels= np.zeros((training_amt*total_words))
    testlabels= np.zeros(((10-training_amt)*total_words))
    
    
    for i in word_list:
        wordptr+=1

        excl = [-1]*10
        for pres in range(10-training_amt):
            while(1):
                nxtrand = np.random.randint(0,10)
                if(excl[nxtrand]==-1):
                    #print("YAHEEEEET "+ str(nxtrand))
                    excl[nxtrand]=nxtrand
                    break
        #print("For " + str(i) + ": " + str(excl))
        for bandnum in range(5):
            ptr2 = 0 #points to which presentation currently (0...9)
            for pres in range(10):
                if(excl[pres]!=-1):
                    continue
               
                TrainingData[bandnum][wordptr][ptr2]=all_data[bandnum][i][pres]
                ptr2+=1
 
        for bandnum in range(5):
            for pres in range(10-training_amt):
                TestingData[bandnum][wordptr][pres] = all_data[bandnum][i][excl[pres]]
   
   
   
   
    simplepair = [category1,category2]
    #words now like cat1, cat1 ,cat1 ... cat1, cat2 ,cat2 ... cat2
    SHheap = []
    TrainWords = []
    TestWords = []

    for band_num in range(5):
        #print(band_num)
        for t in range(tStart, tEnd-tWidth+1, tIncr): #various starts of time slice
            #print(t)
            for channel in range(256):
                featureVectorShTrain= np.zeros((training_amt*total_words,tEx))
                presnum = 0
                for pres in range(training_amt):
                    for wordnum in range(total_words): #pres then word ordering important to make sure ShTrain ends with even in each chunk of 10 (i.e. 1,2,3...10, 1,2,3...10, etc)
                        feature_vector_position = 0
                        
                        trainlabels[presnum]=categoryof[wordnum]
                        
                        for tEStart in range(t,t+tWidth-tEx,tEx): #think abt this one more
                            featureVectorShTrain[presnum][feature_vector_position] = np.average(TrainingData[band_num][wordnum][pres][channel][tEStart:tEStart+int(tWidth/tEx)])
                           
                            feature_vector_position+=1
                        print(featureVectorShTrain[presnum])
                        presnum+=1
 
                #YOIT
                #feature vector should be generated
                flattened = featureVectorShTrain.flatten()
                #first (10 features * 11 words) is first chunk
                chunksz = 10 * total_words
 
                pearsoncoeff = 0
                for i in range(training_amt-1):
                    for j in range(i+1,training_amt):
                        pearsoncoeff += pearsonr(flattened[i*chunksz:(i+1)*chunksz], flattened[j*chunksz:(j+1)*chunksz])[0]
 
                pearsoncoeff /= 28
                if(len(SHheap)<400):
                    heappush(SHheap, (pearsoncoeff, band_num, t, channel, flattened))#push this btc vector
                else:
                    heappushpop(SHheap, (pearsoncoeff, band_num, t, channel, flattened))
 
    #top 400
    toSelect = 10 #how many of the best do we pick
    grandmatrixtrain = np.zeros( (training_amt*total_words, toSelect * tEx) )
    grandmatrixtest = np.zeros( ((10-training_amt)*total_words, toSelect*tEx))
    templist = [] #stores the flattened training vectors
    templist2 = [] #stores BTC parameters for the vectors
    for i in range(400):
        (pearsoncoeff, band_num, t, channel, flatten) = heappop(SHheap)
        if(i>=400-toSelect):
            templist.append(flatten)
            templist2.append( (band_num,t,channel) )
            print(str(pearsoncoeff) + " " + str(band_num) + "  " + str(t) + " "+ str(channel) +" ")
    
    for pres in range(training_amt*total_words):
        thislist = []
        for i in range(len(templist)):
            for j in range(tEx):
                grandmatrixtrain[pres][i*tEx + j] = templist[i][pres*tEx+j]
    
        
    
    #build testing feature matrix and labels
    for pres in range((10-training_amt)*total_words):
        for i in range(toSelect):
            testlabels[(pres%total_words)] = categoryof[(pres%total_words)]
            (band_num, t, channel) = templist2[i]

            ptrO = 0
            for tEStart in range(t,t+tWidth-tEx,tEx):
                grandmatrixtest[pres][i*tEx + ptrO] = np.average(TestingData[band_num][pres%total_words][int(pres/total_words)][channel][tEStart:tEStart+int(tWidth/tEx)])
                ptrO += 1
                

    
    clist = np.logspace(-3,2,100)
    bst_acc = 0
    bst_c = 0
    for c in clist:
        avg_acc = 0
        for fold in range(4):
            fold_size = int(training_amt*total_words/4)
            valid_x = grandmatrixtrain[ (fold_size*fold) : ((fold_size)*(fold+1))]
            valid_y = trainlabels[ (fold_size*fold) : ((fold_size)*(fold+1))]
            tr_x = np.concatenate( (grandmatrixtrain[0:(fold_size*fold)],grandmatrixtrain[(fold+1)*fold_size:training_amt*total_words]), axis=0)
            tr_y = np.concatenate( (trainlabels[0:(fold_size*fold)],trainlabels[(fold+1)*fold_size:training_amt*total_words]), axis=0)

            scaler = StandardScaler()
            tr_x = scaler.fit_transform(tr_x)
            valid_x = scaler.transform(valid_x)
            tr_y = np.ravel(tr_y)
            valid_y = np.ravel(valid_y)

            classifier = LinearSVC(C=c)
            classifier.fit(tr_x,tr_y)
            avg_acc += (classifier.score(valid_x,valid_y))/4.0
        if(avg_acc > bst_acc):
            bst_acc = avg_acc
            bst_c = c
    #print("YET " + str(bst_c))
    classifier = LinearSVC(C=bst_c)
    scaler = StandardScaler()
    grandmatrixtrain = scaler.fit_transform(grandmatrixtrain)
    grandmatrixtest = scaler.transform(grandmatrixtest)
    trainlabels = np.ravel(trainlabels)
    testlabels = np.ravel(testlabels)
    classifier.fit(grandmatrixtrain,trainlabels)
    f.write(str(category1) + " " + str(category2) + " " + str(classifier.score(grandmatrixtest,testlabels)))
    print(str(category1) + " " + str(category2) + " " + str(classifier.score(grandmatrixtest,testlabels)))

    #print(classifier.predict(grandmatrixtest))
    #print(testlabels)

f = open("results_10.29.txt", "w") 
for i in itertools.combinations(range(12),2):
    print("running: " + str(i))
    get_feature_matrix(i[0],i[1])

f.close()
print("finished!")
