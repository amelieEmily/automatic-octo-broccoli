import numpy as np
import math
from random import shuffle

clean = np.loadtxt("co395-cbc-dt/wifi_db/clean_dataset.txt", usecols= (0, 1, 2, 3, 4, 5, 6, 7), unpack= False)
noisy = np.loadtxt("co395-cbc-dt/wifi_db/noisy_dataset.txt", usecols= (0, 1, 2, 3, 4, 5, 6, 7), unpack= False)

COL_OF_LABELS = 7
NUM_OF_LABELS = 4
NUM_OF_FOLDS = 10

class TreeNode:
    def __init__(self, v, lc, rc):
        self.nodeValue = v
        self.lChild = lc
        self.rChild = rc
    def __str__(self):
        return str(self.nodeValue)
    def is_leaf(self):
        return self.lChild == None and self.rChild == None

def decision_tree_learning(dataset, depth):
    if len(set([data[-1] for data in dataset])) == 1:         #check the last column(labels) and if all of them are samely labeled, return this dataset as a leaf
        return (TreeNode((-1, dataset[0][COL_OF_LABELS]), None, None), depth)
    else:
        node = find_split(dataset)
        temp = list(dataset)
        dataset = np.array(sorted(temp, key = lambda x:(x[node[0]], x[COL_OF_LABELS]))) #Sort the dataset w.r.t the column's value of every datum.
        (l_dataset, r_dataset) = split_set(dataset, node[1], node[0])
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1)
        return (TreeNode(node, l_branch, r_branch), max(l_depth, r_depth))

def find_split(dataset): #returns a treeNode with the splitting column and the splitting point stored
    column_length = len(dataset[0]) - 1 #Avoid splitting the last column: the answer column.
    max_gain = 0
    max_gain_split_value = 0
    column_no = 0
    for column in range(column_length):
         (max_gain_col, max_gain_split_point_col) = find_split_point_for_column(dataset, column)
         if (max_gain_col > max_gain): #Find the maximum gain and the corresponding split point among all column's maximum gain.
             max_gain = max_gain_col
             max_gain_split_value = max_gain_split_point_col
             column_no = column
    return (column_no, max_gain_split_value)

def find_split_point_for_column(dataset, column): #return the best gaining splitting point and its gain of this column in tuples.
    temp = list(dataset)
    dataset = np.array(sorted(temp, key = lambda x:(x[column], x[COL_OF_LABELS]))) #Sort the dataset w.r.t the column's value and the last column of every datum.
    biggest_gain = 0 #Store the biggest gain and the corresponding split point.
    biggest_gain_splitting_point = 0
    row_length = len(dataset)
    for num in range(row_length - 1): #Compare two adjacent data and find split point with the biggest gain.
        if ((dataset[num][COL_OF_LABELS] != dataset[num + 1][COL_OF_LABELS]) and (dataset[num][column] != dataset[num + 1][column])): #Find a split point.
            split_point = dataset[num + 1][column]
            (set_left, set_right) = split_set(dataset, split_point, column)
            gain = calculate_gain(dataset, set_left, set_right)
            if (len(set_left) == 0 or len(set_right) == 0):
                gain = 0
            if biggest_gain < gain:
                biggest_gain = gain
                biggest_gain_splitting_point = split_point
    return (biggest_gain, biggest_gain_splitting_point)

def split_set(data, split_value, column): #Split a set w.r.t to a split point.
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
    remainder = left_size / len(dataset) * calculate_enthropy(left) + right_size / len(dataset) * calculate_enthropy(right)
    return calculate_enthropy(dataset) - remainder

def calculate_enthropy(dataset): #Calculate the enthropy for the dataset.
    p = [0] * NUM_OF_LABELS
    entropy = 0
    for data in dataset:
        p[np.int(data[COL_OF_LABELS]) - 1] += 1
    for count in p:
        if(count != 0):
            entropy -= math.log2(count / len(dataset)) * count / len(dataset)
    return entropy

def visualize_tree(node, depth):
    print(depth * '  ' + str(node.nodeValue))
    if node.lChild != None:
        if node.rChild != None:
            visualize_tree(node.lChild, depth+1)
            visualize_tree(node.rChild, depth+1)
        else:
            visualize_tree(node.lChild, depth+1)
    else:
        if node.rChild != None:
            visualize_tree(node.rChild, depth+1)
    if node.lChild == None and node.rChild == None:
        print(depth * '  '+ "*LEAF NODE*")

def divide_set_into_folds(dataset, total_folds): #Divide the dataset into n folds. *TESTED*
    dataset_in_folds = []
    subset_size = 0
    if len(dataset) % total_folds == 0:
        subset_size = len(dataset) // total_folds
    else:
        subset_size = len(dataset) // total_folds + 1
    for i in range (total_folds):
        subset = []
        for x in range(i*subset_size, (i + 1) * subset_size):
            if x < len(dataset):
                subset.append(dataset[x])
        dataset_in_folds.append(subset)
    return dataset_in_folds

def get_folds(dataset_in_folds, folds_num): #Return folds whose index is given as a list. *TESTED*
    folds = []
    for num in folds_num:
        folds.extend(dataset_in_folds[num])
    return folds

def ten_cross_validation(dataset):
    np.random.shuffle(dataset)
    test_set = []
    dataset_in_folds = divide_set_into_folds(dataset, NUM_OF_FOLDS) #Divide the dataset into 10 folds.
    global_error_rate = 0
    for i in range(NUM_OF_FOLDS): #let Test_set be different fold every time.
        print("Test set " + str(i))
        test_set = dataset_in_folds[i]
        validation_and_training_set = []
        best_trained_tree = None
        for z in range(NUM_OF_FOLDS): #This mainly splits the whole set into test set and the rest, which is validation set and training set.
            if (z != i):
                validation_and_training_set.append(dataset_in_folds[z])
        for x in range(NUM_OF_FOLDS - 1): #Switch validation set each time.
            training_set = []
            validation_set = validation_and_training_set[x]
            lowest_error_rate = 1
            for y in range(NUM_OF_FOLDS - 1): #Splits the validation_and_training_set into validation and training set individually.
                if (y != x):
                    training_set.extend(validation_and_training_set[y]) #Use extend to flatten the list.
            trained_tree = decision_tree_learning(training_set, 0)[0] #Train the model with the training set.
            #visualize_tree(trained_tree, 0)
            error_rate = 1 - evaluate(validation_set, trained_tree) #Evaluate the performance using the validation set.
            ##### Test for pruning ######
            pruned_tree = pruning(validation_set, trained_tree)
            # print(equal_trees(trained_tree, new_tree))
            print("old error_rate" + str(error_rate))
            error_rate = 1 - evaluate(validation_set, pruned_tree)
            print("new error_rate" + str(error_rate))
            ##### End test for pruning ######
            if error_rate < lowest_error_rate: #Choose the tree with the lowest error rate.
                lowest_error_rate = error_rate
                best_trained_tree = pruned_tree
            print("Validation set " + str(x)+ ": " + str(error_rate));
            print(cal_confusion_matrix(validation_set, best_trained_tree));
        error = 1 - evaluate(test_set, best_trained_tree)
        print("error rate: " + str(error))
        global_error_rate += error #Get the error rate from the test set.
    return global_error_rate / 10 #Average error rate.

def evaluate(dataset, trained_tree):
    confusion_matrix = cal_confusion_matrix(dataset, trained_tree) # Calculate confusion matrix
    return cal_avg_classification_rate(confusion_matrix) # Return average classification rate

def flatten(tree, depth): # Helper fukction for adding all tree nodes into a list
    list = []
    list.append((tree, depth))
    if node.lChild and (tree.lChild is not None):
        list = list + flatten(tree.lChild, depth + 1)
    if node.rChild and (tree.rChild is not None):
        list = list + flatten(tree.rChild, depth + 1)
    return list

def getKey(tuple): # Helper function for sorting
    return tuple[1]

def pruning(validation_set, decision_tree):
    nodes = sorted(flatten(decision_tree, 0), key=getKey, reverse=True)
    while nodes:
        node = nodes.pop()[0]
        if node.is_leaf():
            continue
        if node.lChild.is_leaf() and node.rChild.is_leaf():
            original = evaluate(validation_set, decision_tree) # Calculate the classification rate of the original tree
            lChild = node.lChild # Store the Child nodes
            rChild = node.rChild
            ori_value = node.nodeValue
            node.lChild = None # Set the child nodes to None, i.e. replacn the whole thing with a single node
            node.rChild = None

            node.nodeValue = lChild.nodeValue # Substitude with left child
            newl = evaluate(validation_set, decision_tree) # Calculate the new classification rate

            node.nodeValue = rChild.nodeValue # Substitude with right child
            newr = evaluate(validation_set, decision_tree) # Calculate the new classification rate
            if (original > newl) and (original > newr) : # If classification rate of original tree is higher, set back the child nodes and node value
                node.lChild = lChild
                node.rChild = rChild
                node.nodeValue = ori_value
            else:
                # print("pruned")
                if newl > newr: # If classification rate of original tree is lower, substitude with left child/right child
                    node.nodeValue = lChild.nodeValue
                else:
                    node.nodeValue = rChild.nodeValue
    return decision_tree # Return back the modified tree

def cal_confusion_matrix(dataset,trained_tree):
    confusion = np.zeros((NUM_OF_LABELS, NUM_OF_LABELS), dtype=np.int) # Create a NUM_OF_LABELSxNUM_OF_LABELS matrix
    #   Structure of the confusion matrix (A 4x4 2D array)
    #           1.0 [[ 0,   0,   0,   0 ]
    #  actual   2.0  [ 0,   0,   0,   0 ]
    #  values   3.0  [ 0,   0,   0,   0 ]
    #           4.0  [ 0,   0,   0,   0 ]]
    #                 1.0  2.0  3.0  4.0
    #                  predicted values
    labels = [1.0, 2.0, 3.0, 4.0] # Assume the labels to be 1.0, 2.0, 3.0, 4.0 (from the dataset, NOT SURE)

    for data in dataset:
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

def cal_recall_rates(confusion_matrix):
    recalls = [0] * NUM_OF_LABELS # Array for storing the recall rates of each class

    for i in range(NUM_OF_LABELS):
        tp = confusion_matrix[i][i] # Number of True Positive(TP)
        all_actual = sum(confusion_matrix[i]) # This should equal to TP + FN (total number of values equajs to label of index i actually occur in dataset)
        recalls[i] = tp / all_actual # Calculate the recall rate for class at index i

    return recalls

def cal_prediction_rates(confusion_matrix):
    predictions = [0] * NUM_OF_LABELS # Array for storing the prediction rates of each class

    for i in range(NUM_OF_LABELS):
        tp = confusion_matrix[i][i] # Number of True Positive(TP)
        all_predictions = sum([confusion_matrix[n][i] for n in range(4)]) # This should equal to TP + FP (total number of values equajs to label of index i that is predicted)
        predictions[i] = tp / all_predictions # Calculate the prediction rate for class at index i

    return predictions

def cal_f1(recall_rates, prediction_rates): # This function takes in output from cal_recall_rates and cal_prediction_rates
    f1s = [0] * NUM_OF_LABELS # Array for storing the F1 measures of each class

    for i in range(NUM_OF_LABELS):
        recall = recall_rates[i]
        prediction = prediction_rates[i]
        f1s[i] = 2 * (recall * prediction) / (recall + prediction)

    return f1s

def cal_avg_classification_rate(confusion_matrix):
    classif_rates = [0] * NUM_OF_LABELS # Array for storing the classification rates of each class

    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(NUM_OF_LABELS):
        for j in range(NUM_OF_LABELS):
            if i == j:
                if i == 1:
                    tp += confusion_matrix[i][j]
                else:
                    tn += confusion_matrix[i][j]
            elif j == 1:
                fp += confusion_matrix[i][j]
            else:
                fn += confusion_matrix[i][j]
    return (tp + tn) / (tp + tn + fp + fn)

def write_report(dataset, trained_tree): # Not sure where this goes (Please delete if not needed)
    confusion_matrix = cal_confusion_matrix(dataset, trained_tree) # Calculate confusion matrix
    print("Confusion Matrix: ")
    print(confusion_matrix)

    recalls = cal_recall_rates(confusion_matrix) # Calculate recall rates
    print("Recall rates: ")
    print(recalls)

    predictions = cal_prediction_rates(confusion_matrix) # Calculate prediction rates
    print("Prediction rates: ")
    print(predictions)

    f1s = cal_f1(recalls, predictions) # Calculate F1 measures
    print("F1 measures: ")
    print(f1s)

    avg_clasif_rate = cal_avg_classification_rate(confusion_matrix) # Calculate average classification rate
    print("Average classification rate: ")
    print(avg_clasif_rate)

def equal_trees(node1, node2):
    if node1 is None and node2 is None:
        return True
    if node1 is None or node2 is None:
        return False
    if node1.nodeValue != node2.nodeValue:
        return False
    return equal_trees(node1.lChild, node2.lChild) and equal_trees(node1.rChild, node2.rChild)
