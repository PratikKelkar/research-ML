import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from heapq import heappush, heappop, heappushpop
import itertools
from scipy.stats import pearsonr
from get_y import get_y


def prf1(test_y, pred):
    tp=0
    fp=0
    tn=0
    fn=0
    for pro in range(len(pred)):
        if(test_y[pro]==1):
            if(pred[pro]==1):
                tp+=1
            if(pred[pro]==0):
                fn+=1
        if(test_y[pro]==0):
            if(pred[pro]==1):
                fp+=1
            if(pred[pro]==0):
                tn+=1

    precision = tp / (tp+fp)  #how accurate positive predictions are
    recall = tp / (tp+fn) #how accurate to classify as positive
    f1 = 2 * (precision*recall) / (precision+recall)
    return (precision,recall,f1)

def finalsvm(totalwords, cat1, cat2):
    np.random.seed(49)
    x = np.load('svm_final_train.npy')
    y = get_y(cat1,cat2)
    jj = 10*totalwords #amount of total data
    tro = int(.8*jj) #amount of training data
    #print(x.shape)
    #print(y.shape)
    train_x = x[:tro]
    train_y = y[:tro]
    test_x = x[tro:jj]
    test_y = y[tro:jj]

    #clist = np.logspace(-2,3,6)
    clist = np.logspace(-3,2,100)
    #clist = [.1,1]
    #print(clist)
    bst_acc = 0
    bst_c = 0
    
    for c in clist:
        avg_acc = 0
        for fold in range(4):
            kk = int(tro/4) #size of each fold
            valid_x = x[(fold*kk):((fold+1)*kk)]
            valid_y = y[(fold*kk):((fold+1)*kk)]

            tr2_x = np.concatenate((x[0:(fold*kk)],x[(fold+1)*kk:tro]),axis=0)
            tr2_y = np.concatenate((y[0:(fold*kk)],y[(fold+1)*kk:tro]),axis=0)

            scaler = StandardScaler()
            tr2_x = scaler.fit_transform(tr2_x)
            valid_x = scaler.transform(valid_x)
            tr2_y = np.ravel(tr2_y)
            valid_y = np.ravel(valid_y)

            classifier = LinearSVC(C=c)
            classifier.fit(tr2_x,tr2_y)
            avg_acc += (classifier.score(valid_x,valid_y))/4.0

        if(avg_acc>bst_acc):
            bst_acc = avg_acc
            bst_c = c
    
    
    savr = []
    
    for c in clist:
        classifier = LinearSVC(C=c)
        classifier.fit(train_x,train_y)
        
        finscore = classifier.score(test_x,test_y)
        print("C = " + str(c) + " | Acc = " + str(finscore))
        savr.append(finscore)
    return savr;
    

            
    
    
