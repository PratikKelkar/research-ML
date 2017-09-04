import matplotlib.pyplot as plt
import numpy as np



for itera in range(5):
    band_to_analyze = itera


    top = np.load("svm_results_" + str(band_to_analyze) + "_abtc.npy");
    top2 = np.load("sh_results_"+str(band_to_analyze)+"_abtc.npy")
    
    jet = plt.get_cmap('winter')
    
    x = []
    y = [] #accuracy vs channel
    z = []
    x2=[]
    y2=[]
    z2=[]
    for row in top:
        y.append(row[0])
        x.append(row[3])
        z.append(row[2])
    for row in top2:
        y.append(row[0])
        x.append(row[3])
        z.append(row[2])
    plt.scatter(x,y,c=z,cmap=jet)
    plt.colorbar()
    plt.show()
    
    plt.scatter(x2,y2,c=z2,cmap=jet)
    plt.colorbar()
    plt.show()
