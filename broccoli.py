import numpy as np
import math

clean = np.loadtxt("co395-cbc-dt/wifi_db/clean_dataset.txt", usecols= (0,1,2,3,4,5,6,7),unpack= False)
noisy = np.loadtxt("co395-cbc-dt/wifi_db/noisy_dataset.txt", usecols= (0,1,2,3,4,5,6,7),unpack= False)

class TreeNode:
    def __init__(self, v, lc, rc):
        self.nodeValue = v
        self.lChild = lc
        self.rChild = rc



# def decision_tree_learning(dataset, depth):
#     if len(set([data[-1] for data in dataset])) == 1:         #check the last column(labels) and if all of them are samely labeled, return this dataset as a leaf
#         return (TreeNode((7,dataset[0][7]), None, None), depth)
#     else:
#
#
# def find_split(dataset):
#     for column in range(7):
#         dataset.sort(key=lambda x:x[column]) #Sort the dataset w.r.t the column's value of every datum.
#         biggest_gain = 0 #Store the biggest gain.
#         for num in range(2000):
#             if dataset[num][7] != dataset[num+1][7]: #Find a split point.
#                 set_left =

def calculate_gain(dataset, left, right):
    left_size = len(left)
    right_size = len(right)
    remainder = left_size/len(dataset) * calculate_enthropy(left) + right_size/len(dataset) * calculate_enthropy(right)
    return calculate_enthropy(dataset) - remainder

def calculate_enthropy(dataset):
    p = []
    dataset.sort(key=lambda x:x[7]) #sort the data based on label
    entropy = 0
    count = 0
    for num in range(len(dataset)-1): #Count the data with same label
        count += 1
        if dataset[num][7] != dataset[num+1][7]: #calculate pk value when a new labeled data is next and push the pk value to the list p
            p.append(count/len(dataset))
            count = 0
        if (num + 2 == len(dataset)): #Edge case: compare the last two and count differently if the two is different
            if (dataset[num][7] == dataset[num+1][7]):
                p.append((count+1)/len(dataset))
            else:
                p.append(count/len(dataset))
                p.append(1/len(dataset))
    for prob in p:
        entropy -= math.log2(prob) * prob
    return entropy
