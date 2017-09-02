import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from heapq import heappush, heappop, heappushpop
import itertools
from scipy.stats import pearsonr
from matnumpy import mat_to_numpy

results_SVM = []    #massive record of all results for each CBTR Vector analyzed using Dalponte's method
results_shinkareva = []  #massive record of all results for each CBTR Vector analyzed using Shinkareva's method

#starting time
tO = 0
#ending time
tN = 650
#width of time slice
tW = 100
#increment in start time
dT = 50
#number of examples to use in sampling procedure, downsample to 100 Hz
tE = 10

#data is organized as follows: a,t,a,t,a,t,a,t,a,t,t (a=animal, t=tool) -> word repeats every 11th time -> animal at 0 is same as animal at 11, 22,etc
#all the way to 110. This ensures that data is not unevenly split into testing, training and validation.
Bands, Y = mat_to_numpy()  #Bands is list of len(5), with Bands[i] being a 110x256x650 matrix    |||| Y is a 110x1 numpy array.

SVMheap = []
svm_save_states = dict()

SHheap = []
sh_save_states = dict()

svm_curr_idx = 0
sh_curr_idx = 0
for bandnum in range(len(Bands)):
    print(bandnum)#5 bands
    allchanneldata = Bands[bandnum]                                                   #all 130 presentations x 256 channels x 750 ms filtered at the appropriate band
    for t in range(tO, (tN-tW), dT):                                    #12 start times (from 100 to 650)
        for channel in range(256):                                    #256 channels
            featureVectorSh = np.zeros((88,10))                #create feature vector, Shinkareva for each channel
            featureVectorSVM = np.zeros((88,30))                  #create feature vector, SVM for each channel
            for presentation in range(88):                                 #88 presentations
                posCounterSh = 0
                posCounterSVM = 0                                                  
                for tEstart in range(t, (t+tW-tE), tE):             #downsample data
                    requiredSegment = allchanneldata[presentation][channel][tEstart:tEstart+10]
                    
                    featureVectorSh[presentation][posCounterSh] = np.average(requiredSegment)        #add 10 vals per presentation
                    posCounterSh += 1
                    
                    featureVectorSVM[presentation][posCounterSVM] = np.average(requiredSegment)
                    posCounterSVM +=1
                    featureVectorSVM[presentation][posCounterSVM] = np.amax(np.absolute(requiredSegment))
                    posCounterSVM+=1
                    featureVectorSVM[presentation][posCounterSVM] = np.amin(np.absolute(requiredSegment))
                    posCounterSVM+=1

            #run svm
            # store (svm result, bandnum, t, channel, featurevector) to results_SVM 
            
            #4 fold cross validation
            average_accuracy_svm=0
            for fold in range(4):
                #includes fold*22...(fold+1)*22-1
                test_x = featureVectorSVM[(fold*22):((fold+1)*22)]
                test_y = Y[(fold*22):((fold+1)*22)]

                train_x = np.concatenate((featureVectorSVM[0:(fold*22)],featureVectorSVM[(fold+1)*22:88]), axis=0)
                train_y = np.concatenate((Y[0:(fold*22)],Y[(fold+1)*22:88]), axis=0)
        
                classifier = LinearSVC()

                scaler = StandardScaler()
                train_x = scaler.fit_transform(train_x)
                test_x = scaler.transform(test_x)
                classifier.fit(train_x,train_y)
                accuracy = classifier.score(test_x,test_y)

                average_accuracy_svm += accuracy/4.0

            svm_save_states[svm_curr_idx] = (bandnum,t,channel,featureVectorSVM)            
            
            if(len(SVMheap) < 400):
                heappush(SVMheap,(average_accuracy_svm,svm_curr_idx))
            else:
                heappushpop(SVMheap,(average_accuracy_svm,svm_curr_idx))
                
            svm_curr_idx+=1

            
            #analyze quality of feature vector using Shinkareva's method
            # store (sh result, bandnum, t, channel) to results_shinkareva

            flattenedVector = featureVectorSh.flatten()
            segmentsToCompare = [flattenedVector[:110], flattenedVector[110:220], flattenedVector[220:330],
                                 flattenedVector[330:440], flattenedVector[440:550], flattenedVector[550:660], flattenedVector[660:770],
                                 flattenedVector[770:880]]

            avgpearson = 0

            for i in range(7):
                for j in range(i+1,8):
                    avgpearson += pearsonr(segmentsToCompare[i], segmentsToCompare[j])[0]
            avgpearson/=28

            sh_save_states[sh_curr_idx] = (bandnum,t,channel,featureVectorSh)

            if(len(SHheap) < 400):
                heappush(SHheap, (avgpearson, sh_curr_idx))
            else:
                heappushpop(SHheap, (avgpearson, sh_curr_idx))

            sh_curr_idx += 1
            
              

svm_results = []
sh_results = []

#print top 400
print("SVM results:")
for i in range(400):
    (svmacc,idx) = heappop(SVMheap)
    print("Band: " + str(svm_save_states[idx][0]) + " | Time: " + str(svm_save_states[idx][1]) + " | Channel: " + str(svm_save_states[idx][2])
          + " | Accuracy: " + str(svmacc))
    svm_results.append((svmacc, svm_save_states[idx][3], svm_save_states[idx][0], svm_save_states[idx][1], svm_save_states[idx][2]))

print("===================================")

print("Pearson results")
for i in range(400):
    (prsn,idx) = heappop(SHheap)
    print("Band: " + str(svm_save_states[idx][0]) + " | Time: " + str(svm_save_states[idx][1]) + " | Channel: " + str(svm_save_states[idx][2]) +
          " | Pearson: " + str(prsn))
    sh_results.append((prsn, sh_save_states[idx][3], sh_save_states[idx][0], sh_save_states[idx][1], sh_save_states[idx][2]))

#save results
np.save("svm_results.npy", np.asarray(svm_results))
np.save("pearson_results.npy", np.asarray(sh_results))

    
