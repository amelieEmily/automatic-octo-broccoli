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

def decision_tree_learning(dataset, depth):
    if len(set([data[-1] for data in dataset])) == 1:         #check the last column(labels) and if all of them are samely labeled, return this dataset as a leaf
        return (TreeNode((7,dataset[0][7]), None, None), depth)
    else:
        node = find_split(dataset)
        temp = list(dataset)
        dataset = np.array(sorted(temp, key=lambda x:(x[node[0]], x[7]))) #Sort the dataset w.r.t the column's value of every datum.
        (l_dataset, r_dataset) = split_set(dataset, node[1], node[0])
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1)
        return (TreeNode(node, l_branch, r_branch), max(l_depth, r_depth))


def find_split(dataset): #returns a treeNode with the splitting column and the splitting point stored
    column_length = len(dataset[0])
    max_gain = 0
    max_gain_split_value = 0
    column_no = 0;
    for column in range(column_length):
         (max_gain_col, max_gain_split_point_col) = find_split_point_for_column(dataset, column)
         if (max_gain_col > max_gain):
             max_gain = max_gain_col
             max_gain_split_value = max_gain_split_point_col
             column_no = column
    return (column_no, max_gain_split_value)

def find_split_point_for_column(dataset, column): #return a best gaining splitting point and its gain in tuples
    temp = list(dataset)
    dataset = np.array(sorted(temp, key=lambda x:(x[column], x[7]))) #Sort the dataset w.r.t the column's value of every datum.
    biggest_gain = 0 #Store the biggest gain.
    biggest_gain_splitting_point = 0
    row_length = len(dataset)
    for num in range(row_length):
        if (dataset[num][7] != dataset[num+1][7]) && (dataset[num][column] != dataset[num+1][column]): #Find a split point.
            split_value = dataset[num+1][column]
            (set_left, set_right) = split_set(dataset, split_value, column)
            gain = calculate_gain(dataset, set_left, set_right)
            if biggest_gain < gain:
                biggest_gain = gain
                biggest_gain_splitting_point = split_value

    return (biggest_gain, biggest_gain_splitting_point)

def split_set(data, split_value, column):
    set_left = []
    set_right = []
    for datum in data:
        if datum[column] < split_value:
            set_left.append(datum)
        else:
            set_right.append(datum)
    return (set_left, set_right)

def calculate_gain(dataset, left, right):
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
