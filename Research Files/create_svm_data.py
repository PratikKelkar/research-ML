import matplotlib.pyplot as plt
import numpy as np
from heapq import heappush, heappop, heappushpop
shres = np.load('sh_results_abtc.npy')
svmres = np.load('svm_results_1_abtc.npy')

shvec = np.load('sh_results_vecs.npy')
svmvec = np.load('svm_results_vecs.npy')

shbandvecs = []
shbandvecs.append(np.load('sh_results_0_vecs.npy'))
shbandvecs.append(np.load('sh_results_1_vecs.npy'))
shbandvecs.append(np.load('sh_results_2_vecs.npy'))
shbandvecs.append(np.load('sh_results_3_vecs.npy'))
shbandvecs.append(np.load('sh_results_4_vecs.npy'))

shbandabtc = []
shbandabtc.append(np.load('sh_results_0_abtc.npy'))
shbandabtc.append(np.load('sh_results_1_abtc.npy'))
shbandabtc.append(np.load('sh_results_2_abtc.npy'))
shbandabtc.append(np.load('sh_results_3_abtc.npy'))
shbandabtc.append(np.load('sh_results_4_abtc.npy'))

svmbandvecs = []
svmbandvecs.append(np.load('svm_results_0_vecs.npy'))
svmbandvecs.append(np.load('svm_results_1_vecs.npy'))
svmbandvecs.append(np.load('svm_results_2_vecs.npy'))
svmbandvecs.append(np.load('svm_results_3_vecs.npy'))
svmbandvecs.append(np.load('svm_results_4_vecs.npy'))

svmbandabtc = []
svmbandabtc.append(np.load('svm_results_0_abtc.npy'))
svmbandabtc.append(np.load('svm_results_1_abtc.npy'))
svmbandabtc.append(np.load('svm_results_2_abtc.npy'))
svmbandabtc.append(np.load('svm_results_3_abtc.npy'))
svmbandabtc.append(np.load('svm_results_4_abtc.npy'))

#cnts = np.zeros( (5,551,257) )


'''
heapo = []
for i in svmres:
    if( cnts[int(i[1]),int(i[2]),int(i[3])] != 0):
        quant = (cnts[int(i[1]),int(i[2]),int(i[3])]+i[0])/2
        if(len(heapo)<10):
            heappush(heapo,(quant,i[1],i[2],i[3]))
        else:
            heappushpop(heapo,(quant,i[1],i[2],i[3]))

for (a,b,c,d) in heapo:
    print(str(b) + " | " + str(c) + " | " + str(d) + " | " + str(a))
'''





grandmatrix = []

#method 1(final acc: 81.8%): output concatenation of top 10 pearson vectors
'''
for pres in range(110):
    thislist = []
    for i in range(390,400):
        for j in shvec[i][pres]:
            thislist.append(j)
    grandmatrix.append(thislist)
'''
#method 2(final acc: 50%): output concatenation of top 10 accuracy vectors
'''
for pres in range(110):
    thislist = []
    for i in range(390,400):
        for j in svmvec[i][pres]:
            thislist.append(j)
    grandmatrix.append(thislist)
'''

#method 3(final acc:59.1%): output concatenation of top 2 pearson from each band
'''
for pres in range(110):
    thislist = []
    for band in range(5):
        for itero in range(398,400):
            for j in shbandvecs[band][itero][pres]:
                thislist.append(j)
    grandmatrix.append(thislist)
'''

#method 4(final acc:72.7%): output concatenation of top 3 pearson in bands 0..2
'''
for pres in range(110):
    thislist = []
    for band in range(3):
        for itero in range(397,400):
            for j in shbandvecs[band][itero][pres]:
                thislist.append(j)
    grandmatrix.append(thislist)
'''
#method 5: top 10 pearsons in band 1
'''
for pres in range(110):
    thislist = []
    for itero in range(390,400):
        for j in shbandvecs[1][itero][pres]:
            thislist.append(j)
    grandmatrix.append(thislist)
'''
#method 6(54.5%): top 10 accuracies in band 1 that are also in top 400 pearson
'''
cnts = np.zeros( (12,256))
for i in shbandabtc[1]:
    
    cnts[int(i[2]/50)][int(i[3])]+=1
currpos = 399;
got = 0
savepos = []
while(currpos>=0):
    thisstuff = svmbandabtc[1][currpos]
    thistime = thisstuff[2]
    thischan = thisstuff[3]
    if(cnts[int(thistime/50)][int(thischan)]>0):
        got+=1
        savepos.append(currpos)
        if(got==10):
            break
    currpos-=1
for pres in range(110):
    thislist = []
    for junk in savepos:
        for j in svmbandvecs[1][junk][pres]:
            thislist.append(j)
    grandmatrix.append(thislist)
'''
#method 7(72.7%): top 10 pearson in band 1 that are also in top 400 accuracies
'''
cnts = np.zeros( (12,256))
for i in svmbandabtc[1]:
    
    cnts[int(i[2]/50)][int(i[3])]+=1
currpos = 399;
got = 0
savepos = []
while(currpos>=0):
    thisstuff = shbandabtc[1][currpos]
    thistime = thisstuff[2]
    thischan = thisstuff[3]
    if(cnts[int(thistime/50)][int(thischan)]>0):
        got+=1
        savepos.append(currpos)
        if(got==10):
            break
    currpos-=1
for pres in range(110):
    thislist = []
    for junk in savepos:
        for j in shbandvecs[1][junk][pres]:
            thislist.append(j)
    grandmatrix.append(thislist)
'''
#method 8: top 5 pearson, top 5 accuracy

for pres in range(110):
    thislist = []
    for itero in range(395,400):
        for j in shvec[itero][pres]:
            thislist.append(j)
        for j in svmvec[itero][pres]:
            thislist.append(j)
    grandmatrix.append(thislist)

np.save('svm_final_train.npy',np.asarray(grandmatrix))
    
