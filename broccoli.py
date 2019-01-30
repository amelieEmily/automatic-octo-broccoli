import numpy as np
import math

clean = np.loadtxt("co395-cbc-dt/wifi_db/clean_dataset.txt", usecols= (0,1,2,3,4,5,6,7),unpack= False)
noisy = np.loadtxt("co395-cbc-dt/wifi_db/noisy_dataset.txt", usecols= (0,1,2,3,4,5,6,7),unpack= False)

class TreeNode:
    def __init__(self, v, lc, rc):
        self.nodeValue = v
        self.lChild = lc
        self.rChild = rc
    def __str__(self):
        return str(self.nodeValue)

def decision_tree_learning(dataset, depth):
    if len(set([data[-1] for data in dataset])) == 1:         #check the last column(labels) and if all of them are samely labeled, return this dataset as a leaf
        return (TreeNode((-1,dataset[0][7]), None, None), depth)
    else:
        node = find_split(dataset)
        temp = list(dataset)
        dataset = np.array(sorted(temp, key=lambda x:(x[node[0]], x[7]))) #Sort the dataset w.r.t the column's value of every datum.
        (l_dataset, r_dataset) = split_set(dataset, node[1], node[0])
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1)
        return (TreeNode(node, l_branch, r_branch), max(l_depth, r_depth))

def find_split(dataset): #returns a treeNode with the splitting column and the splitting point stored
    column_length = len(dataset[0])-1 #Avoid splitting the last column: the answer column.
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
    for num in range(row_length-1):
        if ((dataset[num][7] != dataset[num+1][7]) and (dataset[num][column] != dataset[num+1][column])): #Find a split point.
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

def calculate_gain(dataset, left, right): #Return the gain for the dataset and the given left set and right set.
    left_size = len(left)
    right_size = len(right)
    remainder = left_size/len(dataset) * calculate_enthropy(left) + right_size/len(dataset) * calculate_enthropy(right)
    return calculate_enthropy(dataset) - remainder

def calculate_enthropy(dataset): #Calculate the enthropy for the dataset.
    p = [0,0,0,0]
    entropy = 0
    for data in dataset:
        p[np.int(data[7])-1] += 1
    for prob in p:
        if(prob != 0):
            entropy -= math.log2(prob/len(dataset)) * prob/len(dataset)
    return entropy

def visualize_tree(node, depth):
    print(depth*'  '+str(node.nodeValue))
    if node.lChild != None:
        if node.rChild != None:
            visualize_tree(node.lChild, depth+1)
            visualize_tree(node.rChild, depth+1)
        else:
            visualize_tree(node.lChild, depth+1)
    else:
        if node.rChild != None:
            #print(str(node.rChild))
            visualize_tree(node.rChild, depth+1)
    if node.lChild == None and node.rChild == None:
        print(depth*'  '+"*LEAF NODE*")

def cal_confusion_matrix(test_db,trained_tree):
    confusion = np.zeros((4,4), dtype=np.int) # Create a 4x4 matrix
    #   Structure of the confusion matrix (A 4x4 2D array)
    #           1.0 [[ 0,   0,   0,   0 ]
    #  actual   2.0  [ 0,   0,   0,   0 ]
    #  values   3.0  [ 0,   0,   0,   0 ]
    #           4.0  [ 0,   0,   0,   0 ]]
    #                 1.0  2.0  3.0  4.0
    #                  predicted values
    labels = [1.0, 2.0, 3.0, 4.0] # Assume the labels to be 1.0, 2.0, 3.0, 4.0 (from the dataset, NOT SURE)

    for data in test_db:
        prediction = find_label(data, trained_tree)  #Find the predicted label with the tree
        actual = data[-1]  # The last column is the actual label
        index_actual = labels.index(actual) # Find the index of the actual value in the matrix
        index_predict = labels.index(prediction) # Find the index of the predicted value in the matrix
        confusion[index_actual][index_predict] += 1 # Add one to the correspondind field in the confusion_matrix

    return confusion


def find_label(data, node): # Given a row of data, predict the label it have from the tree node
    if node.nodeValue[0] == -1: # If the node is a leaf node, return label
        return node.nodeValue[1]
    if data[node.nodeValue[0]] < node.nodeValue[1]: # If value at column smaller than split value, recurse on lChild
        return find_label(data, node.lChild)
    else: # If value at column larger than split value, recurse on rChild
        return find_label(data, node.rChild)
