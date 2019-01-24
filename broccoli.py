import numpy as np
import math

clean = np.loadtxt("co395-cbc-dt/wifi_db/clean_dataset.txt", usecols= (0,1,2,3,4,5,6,7),unpack= False)
noisy = np.loadtxt("co395-cbc-dt/wifi_db/noisy_dataset.txt", usecols= (0,1,2,3,4,5,6,7),unpack= False)

temp = clean

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
#         temp = list(dataset)
#         dataset = np.array(sorted(temp,key=lambda x:x[column])) #Sort the dataset w.r.t the column's value of every datum.
#         biggest_gain = 0 #Store the biggest gain.
#         for num in range(2000):
#             if dataset[num][7] != dataset[num+1][7]: #Find a split point.
#                 set_left =

def split_set(data, split_point):
    set_left = []
    set_right = []
    for datum in data:
        if datum < split_point:
            set_left.append(datum)
        else:
            set_right.append(datum)
    return (set_left, set_right)

def calculate_gain(dataset, left, right, column):
    left_size = len(left)
    right_size = len(right)
    remainder = left_size/len(dataset) * calculate_enthropy(left) + right_size/len(dataset) * calculate_enthropy(right)
    return calculate_enthropy(dataset) - remainder

def calculate_enthropy(dataset):
    p = [0,0,0,0]
    entropy = 0
    for data in dataset:
        p[np.int(data[7])-1] += 1
    for prob in p:
        entropy -= math.log2(prob/len(dataset)) * prob/len(dataset)
    return entropy
