import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from heapq import heappush, heappop, heappushpop
import itertools
from scipy.stats import pearsonr
from get_y import get_y

np.random.seed(49)
x = np.load('svm_final_train.npy')
y = get_y()
train_x = x[:88]
train_y = y[:88]
test_x = x[88:110]
test_y = y[88:110]

clist = np.logspace(-3,3,1000)
#clist = [.1,1]
#print(clist)
bst_acc = 0
bst_c = 0
for c in clist:
    avg_acc = 0
    for fold in range(4):
        valid_x = x[(fold*22):((fold+1)*22)]
        valid_y = y[(fold*22):((fold+1)*22)]

        tr2_x = np.concatenate((x[0:(fold*22)],x[(fold+1)*22:88]),axis=0)
        tr2_y = np.concatenate((y[0:(fold*22)],y[(fold+1)*22:88]),axis=0)

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


print(str(bst_acc) + " " + str(bst_c))

train_y = np.ravel(train_y)
test_y = np.ravel(test_y)
#now train and test on entirety
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

classifier = LinearSVC(C=bst_c)
classifier.fit(train_x,train_y)
print(classifier.score(test_x,test_y))
print("Predicted: " + str(classifier.predict(test_x)))
print("Actual   : " + str(test_y))
