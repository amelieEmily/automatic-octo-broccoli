import numpy as np

clean = np.loadtxt("wifi_db/clean_dataset.txt", usecols= (0,1,2,3,4,5,6,7),unpack= False)
noisy = np.loadtxt("wifi_db/noisy_dataset.txt", usecols= (0,1,2,3,4,5,6,7),unpack= False)

class TreeNode:
    nodeValue = (-1, -1)
    lChild = None
    rChild = None

def decision_tree_learning(dataset, depth):
    if len(set([data[-1] for data in dataset])) == 1:
        return()
