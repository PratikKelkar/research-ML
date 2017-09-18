import matplotlib.pyplot as plt
import numpy as np
from heapq import heappush, heappop, heappushpop
def create_svm_data(totalwords):
    shres = np.load('sh_results_abtc.npy')
    svmres = np.load('svm_results_abtc.npy')

    shvec = np.load('sh_results_vecs.npy')
    svmvec = np.load('svm_results_vecs.npy')

    grandmatrix = []

    #method 1(final acc: 86.4%): output concatenation of top 10 pearson vectors

    for i in range(390,400):
        print(str(shres[i][0]) + " " + str(shres[i][1]) + " " + str(shres[i][2]) + " " + str(shres[i][3]))

    for pres in range(10*totalwords):
        thislist = []
        for i in range(390,400):
            for j in shvec[i][pres]:
                thislist.append(j)
        grandmatrix.append(thislist)
    
    #method 2(final acc: 50%): output concatenation of top 10 accuracy vectors
    '''
    for pres in range(10*totalwords):
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
    '''
    for pres in range(10*totalwords):
        thislist = []
        for itero in range(390,400):
            for j in shvec[itero][pres]:
                thislist.append(j)
            for j in svmvec[itero][pres]:
                thislist.append(j)
        grandmatrix.append(thislist)
    '''
    np.save('svm_final_train.npy',np.asarray(grandmatrix))

